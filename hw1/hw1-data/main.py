import pandas as pd
import numpy as np
import subprocess
import time

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from collections import Counter

TRAIN_PATH = "income.train.5k.csv"
DEV_PATH = "income.dev.csv"
TEST_PATH = "income.test.blind.csv"
OUTPUT_PATH = "income.test.predicted.csv"
ERROR_LOG_PATH = "dev_error_log.txt"

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    dev_df = pd.read_csv(DEV_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, dev_df, test_df

def transform(train, dev, test):
    # Separate features and target
    X_train, y_train = train.drop(columns=['id', 'target']), train['target']
    X_dev, y_dev = dev.drop(columns=['id', 'target']), dev['target']
    X_test = test.drop(columns=['id'], errors='ignore')  # Drop 'id' if present

    # Identify feature types
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = ['age', 'hours']

    # Define preprocessing pipeline
    transformer = ColumnTransformer([
        ('num', MinMaxScaler(feature_range=(0, 2)), numerical_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ])

    # Apply transformations
    X_train_enc = transformer.fit_transform(X_train)
    X_dev_enc = transformer.transform(X_dev)
    X_test_enc = transformer.transform(X_test)

    return X_train_enc, y_train, X_dev_enc, y_dev, X_test_enc

def knn_classifier(x_train, y_train, x_query, k):
    # Compute Manhattan distances in a vectorized way
    distances = np.sum(np.abs(x_query[:, np.newaxis] - x_train), axis=2) 

    # Get indices of k nearest neighbors
    nearest_indices = np.argsort(distances, axis=1)[:, :k]

    # Predict labels
    preds = []
    for row in nearest_indices:
        labels = y_train[row]
        most_common = Counter(labels).most_common(1)[0][0]
        preds.append(most_common)

    return preds

def predict(x_train, y_train_raw, x_dev, y_dev_raw, x_test):
    import numpy as np
    import html

    print("Precomputing distances...")

    # Decode HTML entities
    y_train_raw = [html.unescape(label) for label in y_train_raw]
    y_dev_raw = [html.unescape(label) for label in y_dev_raw]

    # Map labels to binary
    label_map = {'<=50K': 0, '>50K': 1}
    y_train = np.array([label_map[label] for label in y_train_raw])
    y_dev = np.array([label_map[label] for label in y_dev_raw])

    # Precompute Manhattan distances
    train_distances = np.sum(np.abs(x_train[:, np.newaxis] - x_train), axis=2)
    dev_distances = np.sum(np.abs(x_dev[:, np.newaxis] - x_train), axis=2)
    test_distances = np.sum(np.abs(x_test[:, np.newaxis] - x_train), axis=2)

    # Precompute sorted indices
    train_sorted_idx = np.argsort(train_distances, axis=1)
    dev_sorted_idx = np.argsort(dev_distances, axis=1)
    test_sorted_idx = np.argsort(test_distances, axis=1)

    best_k = None
    best_dev_error = float('inf')

    print("Searching for best k...")
    for k in range(1, 100, 2):
        # Predict using top-k neighbors
        train_preds = [np.bincount(y_train[idx[:k]]).argmax() for idx in train_sorted_idx]
        dev_preds = [np.bincount(y_train[idx[:k]]).argmax() for idx in dev_sorted_idx]

        # Compute errors and positive prediction rates
        train_error = 100 * np.mean(train_preds != y_train)
        dev_error = 100 * np.mean(dev_preds != y_dev)
        train_pos = 100 * np.mean(np.array(train_preds) == 1)
        dev_pos = 100 * np.mean(np.array(dev_preds) == 1)

        if dev_error < best_dev_error:
            best_k = k
            best_dev_error = dev_error
            best_train_error = train_error
            best_train_pos = train_pos
            best_dev_pos = dev_pos

        print(f"k={k:3d}  train_err={train_error:.1f}% (+:{train_pos:.1f}%)  dev_err={dev_error:.1f}% (+:{dev_pos:.1f}%)")

    print(f"✅ Best k={best_k}  train_err={best_train_error:.1f}% (+:{best_train_pos:.1f}%)  dev_err={best_dev_error:.1f}% (+:{best_dev_pos:.1f}%)")

    print("Predicting on test set...")
    y_test_binary = [np.bincount(y_train[idx[:best_k]]).argmax() for idx in test_sorted_idx]

    # Convert back to string labels
    reverse_map = {0: '<=50K', 1: '>50K'}
    y_test = [reverse_map[label] for label in y_test_binary]

    return y_test, best_dev_error

def write_out(y_test, err_rate, test_path=TEST_PATH, output_path=OUTPUT_PATH, log_path=ERROR_LOG_PATH):
    test = pd.read_csv(test_path)
    test = test.copy()
    test["target"] = y_test
    test.to_csv(output_path, index=False)

    # Log the dev error
    with open(log_path, 'a') as f:
        f.write(f"{err_rate:.2f}\n")

def validate():
    subprocess.run(["python", "validate.py", OUTPUT_PATH])

def main():
    start_time = time.time()

    print("🔹 Loading data...")
    train, dev, test = load_data()

    print("🔹 Preprocessing data...")
    x_train, y_train, x_dev, y_dev, x_test = transform(train, dev, test)

    print("🔹 Finding best predictions...")
    y_test, err_rate = predict(x_train, y_train, x_dev, y_dev, x_test)

    # Compute positive percentage on test set
    test_pos_rate = 100 * np.mean([label == '>50K' for label in y_test])
    print(f"Test set positive prediction rate: {test_pos_rate:.1f}%")


    print("🔹 Writing out...")
    write_out(y_test, err_rate)

    print("🔹 Validating results...")
    validate()

    elapsed = time.time() - start_time
    print(f"✅ Completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()