import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


class NN(Model):
    """MLP with specified number of hidden layers. Dropout after each hidden
    optional always-on dropout."""

    def __init__(self,
                 no_params: int,
                 neurons: list,
                 dropout_probs: list,
                 always_on_dropout: bool = False):
        super(NN, self).__init__(name='NN')  # inherits the parent class and names it (for debugging)

        self.neurons = neurons
        self.no_params = no_params  # number of parameters we output
        self.always_on_dropout = always_on_dropout  # if true then we leave on for test passes

        # Define our layers - we will loop over creating the required hidden and dropout layers
        self.norm = tf.keras.layers.Normalization(axis=-1, name="input_norm")

        self.hidden_layers = []

        for size, p in zip(neurons, dropout_probs):
            self.hidden_layers.append(tf.keras.layers.Dense(size, activation="relu"))
            self.hidden_layers.append(tf.keras.layers.Dropout(rate=p))

        self.out = Dense(no_params, activation="linear", name="out")  # Applies linear activation

    def call(self,
             inputs: tf.Tensor,
             training: bool = False) -> tf.Tensor:
        """Forward pass with dropout flag logic."""
        dropout_flag = training or self.always_on_dropout

        # Apply our custom layers
        x = self.norm(inputs)
        # every second layer is a Dropout, so we toggle `training` only on those
        for layer in self.hidden_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training=dropout_flag)
            else:
                x = layer(x)
        return self.out(x)


class GatedNN(Model):
    """
    Same MLP as before, but the last layer is split into
    (1) a sigmoid gate g_j  ∈ (0, 1) that predicts “is θ_j non-zero?”
    (2) a linear value v_j  that predicts the magnitude if it *is* non-zero.

    The final estimate is         θ̂_j = g_j · v_j
    so if g_j→0 the output is forced to 0.
    """

    def __init__(self,
                 no_params: int,
                 neurons: list,
                 dropout_probs: list,
                 always_on_dropout: bool = False):
        super().__init__(name="GatedNN")

        self.no_params = no_params
        self.always_on_dropout = always_on_dropout

        self.norm = tf.keras.layers.Normalization(axis=-1, name="input_norm")

        self.hidden_layers = []
        for size, p in zip(neurons, dropout_probs):
            self.hidden_layers.append(Dense(size, activation="relu"))
            self.hidden_layers.append(Dropout(rate=p))

        # the two parallel heads
        self.gate_head = Dense(no_params, activation="sigmoid", name="gate")
        self.value_head = Dense(no_params, activation="linear", name="value")

    def call(self, inputs, training=False):
        dropout_flag = training or self.always_on_dropout
        x = self.norm(inputs)
        for layer in self.hidden_layers:
            x = layer(x, training=dropout_flag) if isinstance(layer, Dropout) else layer(x)

        g = self.gate_head(x)  # probability of *not* being zero
        v = self.value_head(x)  # magnitude if non-zero
        theta_hat = g * v  # element-wise product
        return tf.concat([theta_hat, g], axis=1)  # we must return both for the custom loss


class ThetaOnly(tf.keras.Model):
    """
    Thin wrapper that converts a *gated* core model into a plain
    “theta-hat–only” predictor.

    Many evaluation utilities assume the model’s output shape matches the
    ground-truth θ (shape = ``(batch, p)``).  Gated architectures, however,
    often return extra information such as per-dimension gate probabilities
    or concatenate those gates to the main output.  `ThetaOnly` hides that
    complexity by:

    1. Delegating the forward pass to the supplied *core_model*.
    2. Stripping everything but the estimated coefficients θ̂.
       * tuple / list → returns element 0
       * dict         → returns value under key ``"theta"``
       * single tensor twice as wide as ``p`` → returns the first ``p`` columns

    With this wrapper you can pass the resulting instance to legacy metrics
    like *average relative error* or any code that expects
    ``model(x) → (batch, p)``.

    Parameters
    ----------
    core_model : tf.keras.Model
        The trained gated model whose first output (or ``"theta"`` key) is
        θ̂ and whose total number of true parameters is accessible as
        the attribute ``no_params``.
    """
    def __init__(self, core_model):
        super().__init__()
        self.core = core_model
        self.p = getattr(core_model, "no_params", None)

    def call(self, x, training=False):
        y_pred = self.core(x, training=training)
        if isinstance(y_pred, (list, tuple)):
            return y_pred[0]              # θ̂
        if isinstance(y_pred, dict):
            return y_pred["theta"]        # θ̂
        return y_pred[:, :self.p]
