from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return metrics


def print_metrics(metrics):
    print("Model Performans Metrikleri:")
    print(f"Doğruluk (Accuracy): {metrics['accuracy']:.2f}")
    print(f"Hassasiyet (Precision): {metrics['precision']:.2f}")
    print(f"Geri Çağırma (Recall): {metrics['recall']:.2f}")
    print(f"F1 Skoru: {metrics['f1_score']:.2f}")


if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data, split_data
    from model_training import train_svm

    data_path = '//SVM/data/data.csv'
    data = load_data(data_path)
    X, y = preprocess_data(data)
    if X is not None and y is not None:
        X_train, X_test, y_train, y_test = split_data(X, y)
        model = train_svm(X_train, y_train, kernel='linear')
        metrics = evaluate_model(model, X_test, y_test)
        print_metrics(metrics)
