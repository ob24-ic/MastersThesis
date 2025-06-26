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
