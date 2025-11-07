#!/usr/bin/env python3
"""
Usage:
    python knn.py --train <train_file_path> --dev <dev_file_path> -k <number_of_neighbors> [-p <order_of_norm>] [--binarization_method <binarization_method>]
    OR
    python knn.py --train <train_file_path> --test <test_file_path> -k <number_of_neighbors> [-p <order_of_norm>] [--binarization_method <binarization_method>]

    - <train_file_path> : Path to training file. This argument is required.
    - <dev_file_path>   : Path to development file. This argument is mutually exclusive with --test.
    - <test_file_path>  : Path to test file. This argument is mutually exclusive with --dev.
                        In --test mode, the output file will be saved with the same path and name as the test file, but with ".predicted" added to the filename.
    - <number_of_neighbors> : Specify the number of neighbors (k) to consider. This argument is required.
    - <order_of_norm> : Specify the order of the norm (p) to use for distance calculations.
                        Default is 2, which corresponds to the Euclidean distance.
                        p=1 corresponds to the Manhattan distance.
                        This argument is optional.
    - <binarization_method> : Specify the binarization method to use.
                        Default is "Smart+Scaling".
                        Choices are "Naive", "Smart", and "Smart+Scaling".
                        This argument is optional.

Examples:
    1. To use the train and dev files with Manhattan distance:
        python knn.py --train "hw1-data/income.train.txt.5k" --dev "hw1-data/income.dev.txt" -k 41 -p 1
        Output:
            k=41   train_err 17.4% (+: 21.1%) dev_err 14.1% (+: 20.5%)

    2. To use the train and test files with Euclidean distance:
        python knn.py --train "hw1-data/income.train.txt.5k" --test "hw1-data/income.test.blind" -k 41 -p 2
        Output:
            k=41   train_err 17.4% (+: 20.9%) test_pos (+: 20.9%)
            Predictions saved to "hw1-data/income.test.blind.predicted"
"""
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

pos_label = '>50K'

def read_data(filename, use_target=False):
    column_names = ["age", "sector", "edu", "marriage", "occupation", "race", "gender", "hours", "country"]
    if use_target:
        column_names += ["target"]
    data = pd.read_csv(filename, sep=", ", names=column_names, engine='python')

    if use_target:
        X = data.drop("target", axis=1)
        Y = data["target"] == pos_label # bool array
    else:
        X = data
        Y = None

    return X, Y

def fit_encoder(data_X, data_Y, binarization_method):
    if binarization_method == "Naive":
        encoder_X = OneHotEncoder(sparse=False, handle_unknown='ignore')
    else:
        num_processor = MinMaxScaler(feature_range=(0, 2)) if "Scaling" in binarization_method else "passthrough"
        cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')

        encoder_X = ColumnTransformer([
            ("num", num_processor, ["age", "hours"]),
            ("cat", cat_processor, ["sector", "edu", "marriage", "occupation", "race", "gender", "country"])
        ])
    encoder_X.fit(data_X)
    
    return encoder_X

def process(args, train_data, encoder_X, is_dev=True):
    if is_dev:
        X_df, Y_df = read_data(args.dev, use_target=True)
    else:
        X_df, _ = read_data(args.test, use_target=False)

    X = encoder_X.transform(X_df)
    if is_dev:
        Y = Y_df.to_numpy()

    train_X, train_Y = train_data
    train_err, train_pos = eval(args.k, train_X, train_Y, train_data, args.p)

    if is_dev:
        dev_err, dev_pos = eval(args.k, X, Y, train_data, args.p)
        print("k=%-4d train_err %4.1f%% (+:%5.1f%%) dev_err %4.1f%% (+:%5.1f%%)" % (args.k, train_err, train_pos, dev_err, dev_pos))
    else:
        test_Y_pred = predict(args.k, X, train_data, args.p)
        test_pos = sum(test_Y_pred) / len(X) * 100
        print("k=%-4d train_err %4.1f%% (+:%5.1f%%) test_pos (+:%5.1f%%)" % (args.k, train_err, train_pos, test_pos))
        save_to_file(args.test, test_Y_pred)

def knn(k, example, train, ord):
    trainX, trainY = train
    if k < len(trainX):
        distances = np.linalg.norm(example - trainX, axis=1, ord=ord)
        neighbors = np.argpartition(distances, k)[:k]
        votes = trainY[neighbors] # slicing
    else:
        votes = trainY # not enough votes: take all training data
    return np.sum(votes) > len(votes) / 2 # majority vote: more than half

def eval(k, dev_X, dev_Y, train, ord):
    pred = predict(k, dev_X, train, ord)
    errors = sum(pred != dev_Y)
    positives = sum(pred)
    return errors / len(dev_X) * 100, positives / len(dev_X) * 100

def predict(k, test_X, train, ord):
    pred = np.array([knn(k, vecx, train, ord) for vecx in test_X])
    return pred

def save_to_file(file_path, predictions):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    if len(predictions) != len(lines):
        raise ValueError("Number of predictions must match the number of rows in the file")

    new_file_name = file_path + '.predicted'
    with open(new_file_name, 'w') as f:
        for i, features in enumerate(lines):
            f.write(features + ", " + predictions[i] + '\n')

    print(f"Predictions saved to {new_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to training file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dev", type=str, help="Path to development file")
    group.add_argument("--test", type=str, help="Path to test file")
    parser.add_argument("-k", type=int, required=True,
                        help="Specify the number of neighbors (k) to consider."
    )
    parser.add_argument("-p", type=int, default=2,
                        help="Specify the order of the norm (p) to use for distance calculations. Default is 2, which corresponds to the Euclidean distance."
    )
    parser.add_argument("--binarization_method", type=str,
        default="Smart+Scaling",
        help="The binarization method to use.",
        choices=[
            "Naive", "Smart", "Smart+Scaling"
        ]
    )

    args = parser.parse_args()

    train_X_df, train_Y_df = read_data(args.train, use_target=True)
    encoder_X = fit_encoder(train_X_df, train_Y_df, args.binarization_method)
    train_X = encoder_X.transform(train_X_df)
    train_Y = train_Y_df.to_numpy()
    train_data = train_X, train_Y

    if args.dev:
        process(args, train_data, encoder_X, is_dev=True)
    elif args.test:
        process(args, train_data, encoder_X, is_dev=False)