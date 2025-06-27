import numpy as np


def evaluate_sparsity_recovery(model, simulate_batch, threshold=0.05, batch_size=1000, mc_samples=30):
    """
    Evaluate sparsity recovery using false positive/negative rates.

    Args:
        model: trained Keras model
        simulate_batch: function returning (summaries, theta_true)
        threshold: abs(theta_pred) below this is considered zero
        batch_size: number of test samples
        mc_samples: MC dropout samples

    Returns:
        dict with FPR, FNR, precision, recall, f1
    """
    summaries, theta_true = simulate_batch(batch_size=batch_size)
    theta_true_np = theta_true.numpy().squeeze(-1)  # (N, D)

    preds = np.stack([
        model.predict(summaries, verbose=0) for _ in range(mc_samples)
    ])  # (K, N, D)

    theta_pred_mean = preds.mean(axis=0)  # (N, D)

    pred_support = np.abs(theta_pred_mean) > threshold
    true_support = theta_true_np != 0

    # Flatten to compute rates
    pred_flat = pred_support.flatten()
    true_flat = true_support.flatten()

    TP = np.sum(pred_flat & true_flat)
    FP = np.sum(pred_flat & ~true_flat)
    FN = np.sum(~pred_flat & true_flat)
    TN = np.sum(~pred_flat & ~true_flat)

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    fpr = FP / (FP + TN + 1e-8)
    fnr = FN / (FN + TP + 1e-8)

    return {
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "precision": precision,  # True positives / (True positives + False Positives)
        "recall": recall,  # True positives / (True positives + False Negatives)
        "f1_score": f1  # 2 * (Precision * Recall) / (Precision + Recall)
    }
