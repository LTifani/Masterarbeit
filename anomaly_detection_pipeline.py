import json
import torch
import optuna
import numpy as np
from models import Model
from utils import ic, timer, Config
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, precision_score


# Default hyperparameters for the model
HYPERPARAMS = {
    "filters": 16,
    "kernel_size": 2,
    "dilations": (1, 2, 4, 8),
    "dropout": 0.2,
    "latent_dim": 8,
    "batch_size": 16,
    "epochs": 2,
    "learning_rate": 1e-3,
    "early_stopping": True,
    "patience": 8,
}


def load_dataset(npz_path):
    """
    Load the dataset from a .npz file.

    Args:
        npz_path (str): Path to the .npz file containing the preprocessed data.

    Returns:
        tuple: A tuple containing:
            - X_train (np.ndarray): Training data features.
            - splits (dict): Dictionary containing validation and test splits.
            - meta (dict): Metadata associated with the dataset.
    """
    data = np.load(npz_path, allow_pickle=True)
    X_train = data['X_train']  # (N_train, T, 1)
    splits = {
        'val': {
            'normal': data['X_val_normal'],
            'abn_plus_mixed': data['X_val_abnormal'],  # Positive class
            # 'abn_plus_mixed': data['X_val_abn_plus_mixed'],  # Positive class
        },
        'test': {
            'normal': data['X_test_normal'],
            # 'abn_plus_mixed': data['X_test_abn_plus_mixed'],
            'abn_plus_mixed': data['X_test_abnormal'],
        }
    }
    meta = json.loads(data['meta'].item())
    return X_train, splits, meta


def compute_threshold_from_normal_errors(errors_normal):
    """
    Compute the anomaly threshold based on normal data errors.

    The threshold is calculated as mean + 3 standard deviations.

    Args:
        errors_normal (np.ndarray): Reconstruction errors for normal data.

    Returns:
        float: The calculated threshold value.
    """
    mean_error = np.mean(errors_normal)
    std_error = np.std(errors_normal)
    return mean_error + 3 * std_error


def compute_anomaly_detection_metrics(errors, true_labels, threshold, k=10):
    """
    Compute various metrics for anomaly detection based on reconstruction errors.

    Args:
        errors (np.ndarray): Array of reconstruction errors.
        true_labels (np.ndarray): Array of true binary labels (0 for normal, 1 for anomaly).
        threshold (float): Threshold value to classify anomalies.
        k (int): Number of top-K samples to consider for Precision@K.

    Returns:
        dict: A dictionary containing AUC-ROC, F1 score, Precision@K, Precision, and Recall.
    """
    predicted_labels = (errors > threshold).astype(int)

    auc_roc = roc_auc_score(true_labels, errors)  # Use raw errors for AUC
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary', zero_division=0)

    # Precision@K: Sort errors descending, take top-K as anomalies
    sorted_indices = np.argsort(errors)[::-1]
    top_k_predictions = np.zeros_like(true_labels)
    top_k_predictions[sorted_indices[:k]] = 1
    precision_at_k = precision_score(true_labels, top_k_predictions)

    return {
        'AUC-ROC': auc_roc,
        'F1': f1,
        'Precision@K': precision_at_k,
        'Precision': precision,
        'Recall': recall
    }

def optuna_objective(trial: "optuna.Trial", X_train, X_val_normal, seq_len, n_feat, epochs=5):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.Trial): Optuna trial object.
        X_train (np.ndarray): Training data.
        X_val_normal (np.ndarray): Normal validation data.
        seq_len (int): Sequence length of the time series.
        n_feat (int): Number of features in the time series.
        epochs (int): Number of epochs for training during optimization.

    Returns:
        float: The validation loss to minimize.
    """
    # Suggest hyperparameters for this trial
    suggested_filters = trial.suggest_int('filters', 16, 128)
    suggested_kernel_size = trial.suggest_int('kernel_size', 3, 7, step=2)  # Odd kernels
    suggested_dropout = trial.suggest_float('dropout', 0.0, 0.5)
    suggested_learning_rate = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    suggested_batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Consolidate hyperparameters into a dictionary
    suggested_hyperparameters = {
        "epochs": 2,
        "patience": 5,
        "latent_dim": 10,
        "filters": suggested_filters,
        "dropout": suggested_dropout,
        "learning_rate": suggested_learning_rate,
        "early_stopping": True,
        "batch_size": suggested_batch_size,
        "dilations": (1, 2, 4, 8),
        "kernel_size": suggested_kernel_size,
    }

    config = Config()
    config.train_hyperparameter = suggested_hyperparameters

    # Prepare datasets and dataloaders
    train_dataset = TensorDataset(torch.from_numpy(X_train.astype(np.float32)))
    val_dataset = TensorDataset(torch.from_numpy(X_val_normal.astype(np.float32)))
    val_loader = DataLoader(val_dataset, suggested_batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, suggested_batch_size, shuffle=True)

    # Initialize and train the model
    model = Model(seq_len, n_feat, train_loader, val_loader, config, filters=suggested_filters, kernel_size=suggested_kernel_size, dropout=suggested_dropout)
    validation_loss = model.train_model(False)

    ic(suggested_filters, suggested_kernel_size, suggested_dropout, suggested_learning_rate, suggested_batch_size, validation_loss)
    return validation_loss


def run_anomaly_detection_pipeline(npz_path='preprocess/emg_phase1_per_segment.npz', epochs=1, n_trials=1):
    """
    Main pipeline for anomaly detection using hyperparameter optimization.

    Args:
        npz_path (str): Path to the preprocessed data file.
        epochs (int): Number of epochs for the final model training.
        n_trials (int): Number of Optuna trials for hyperparameter optimization.
    """
    

    # 1. Load Data
    X_train, splits, meta = load_dataset(npz_path)
    seq_len, n_feat = X_train.shape[1], X_train.shape[2]

    X_val_normal = splits['val']['normal'].astype(np.float32)
    X_val_positive = splits['val']['abn_plus_mixed'].astype(np.float32)
    X_val = np.concatenate([X_val_normal, X_val_positive], axis=0)
    y_val = np.concatenate([np.zeros(len(X_val_normal)), np.ones(len(X_val_positive))])

    X_test_normal = splits['test']['normal'].astype(np.float32)
    X_test_positive = splits['test']['abn_plus_mixed'].astype(np.float32)
    X_test = np.concatenate([X_test_normal, X_test_positive], axis=0)
    y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_test_positive))])

    # 2. Optuna HPO
    study = optuna.create_study(direction='minimize')  # Minimize val loss
    objective = lambda trial: optuna_objective(trial, X_train, X_val_normal, seq_len, n_feat, epochs)
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)

    # 3. Train final model with best params
    best_batch_size = best_params['batch_size']
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val.astype(np.float32))), batch_size=best_batch_size, shuffle=False)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train.astype(np.float32))), batch_size=best_batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test.astype(np.float32))), batch_size=best_batch_size, shuffle=False)
    val_normal_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_normal.astype(np.float32))), batch_size=best_batch_size, shuffle=False)

    config = Config()
    config.train_hyperparameter = {**HYPERPARAMS, **best_params}
    model = Model(seq_len, n_feat, train_loader, val_loader, filters=best_params['filters'], kernel_size=best_params['kernel_size'], dropout=best_params['dropout'])
    model.train_model(False)

    # 4. Compute Threshold on Normal Validation Data
    errors_val_normal = model.compute_reconstruction_errors(val_normal_loader)
    threshold = compute_threshold_from_normal_errors(errors_val_normal)

    # 5. Evaluate on Validation
    errors_val = model.compute_reconstruction_errors(val_loader)
    val_metrics = compute_anomaly_detection_metrics(errors_val, y_val, threshold, k=10)
    print("Validation Metrics:", val_metrics)

    # 6. Test on Test Set
    errors_test = model.compute_reconstruction_errors(test_loader)
    test_metrics = compute_anomaly_detection_metrics(errors_test, y_test, threshold, k=10)
    print("Test Metrics:", test_metrics)

    # Save model (optional)
    model.save_best_model()


if __name__ == "__main__":
    run_anomaly_detection_pipeline()