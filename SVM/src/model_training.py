from sklearn.svm import SVC
import joblib


def train_svm(X_train, y_train, kernel='linear'):
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    return model


def save_model(model, file_path='models/svm_model.pkl'):
    joblib.dump(model, file_path)
    print(f"Model '{file_path}' olarak kaydedildi.")


if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data, split_data

    data_path = '//SVM/data/data.csv'
    data = load_data(data_path)
    X, y = preprocess_data(data)
    if X is not None and y is not None:
        X_train, X_test, y_train, y_test = split_data(X, y)
        model = train_svm(X_train, y_train, kernel='linear')
        save_model(model, '//SVM/models/svm_model.pkl')
