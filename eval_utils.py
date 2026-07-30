
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, accuracy_score

def get_split(X, y, stratify=None, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=seed)

def regression_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return {"RMSE": round(rmse, 3), "MAE": round(mae, 3)}

def classification_metrics(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Macro_F1": round(f1_score(y_true, y_pred, average="macro"), 3)
    }
