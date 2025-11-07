from __future__ import division

import pandas as pd
import time
import os
from svector import svector
from collections import Counter
import random

__author__ = "Protzela"

TRAIN_PATH = "train.csv"
DEV_PATH = "dev.csv"
TEST_PATH = "test.csv"
OUTPUT_PATH = "test.predicted.csv"
LOG_PATH = "log.txt"

EPOCHS = list(range(1, 100, 5))

def symbol(error_rate):
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w') as f:
            f.write(f"{error_rate}\n")

    with open(LOG_PATH, 'r') as file:
        lines = file.readlines()

    last = float(lines[-1].strip())
    best = min(float(line.strip()) for line in lines)

    s1 = '=' if error_rate == last else ('-' if error_rate > last else '+')
    s2 = '=' if error_rate == best else ('-' if error_rate > best else '+')

    return s1 + s2

def read_from(textfile):
    data = pd.read_csv(textfile)
    for i in range(len(data)):
        id, words, label = data.iloc[i]
        yield (1 if label=="+" else -1, words.split())

def dictionary():
    counter = Counter()
    for i, words in read_from(TRAIN_PATH):
        for ii, word in enumerate(words):
            counter[word] += 1
            if ii < len(words) - 1:
                bigram = f"{word}_{words[ii+1]}"
                counter[bigram] += 1
    book = {word for word, count in counter.items() if count > 1}
    print(f"   🔸 Dropped {len(counter) - len(book)} one-count features")
    return book

def make_vector(words, book):
    v = svector()
    for i, word in enumerate(words):
        if word in book:
            v[word] += 1
        if i < len(words) - 1:
            bigram = f"{word}_{words[i+1]}"
            if bigram in book:
                v[bigram] += 1
    v["<bias>"] = 1
    return v

def test(model, book, file_path):
    pred = []
    err = 0

    for i, (label, words) in enumerate(read_from(file_path), 1):
        x = make_vector(words, book)
        score = model.dot(x)
        pred_label = 1 if score > 0 else -1
        pred.append("+" if pred_label == 1 else "-")
        err += pred_label != label

    return pred, err / i

def train(book):
    start_time = time.time()
    best_model = None
    best_dev_error = 1.0
    best_epoch = 0

    for E in EPOCHS:
        model = svector()
        model_sum = svector()
        seen = 0

        for epoch in range(E):
            examples = list(read_from(TRAIN_PATH))
            random.shuffle(examples)

            for label, words in examples:
                x = make_vector(words, book)
                if label * model.dot(x) <= 0:
                    model += label * x
                    model_sum += seen * (label * x)
                seen += 1

        model_avg = svector()
        for f in model:
            model_avg[f] = model[f] - (model_sum.get(f, 0) / seen)

        _, dev_error = test(model_avg, book, DEV_PATH)

        print(f"   🔸 Epochs={E}, Dev err={dev_error * 100:.3f}%")

        if dev_error < best_dev_error:
            best_dev_error = dev_error
            best_model = model_avg
            best_epoch = E

    print("   🔸 Best dev err %.1f%% at %d epochs, |model|=%d, time: %.1f secs" %
          (best_dev_error * 100, best_epoch, len(best_model), time.time() - start_time))
    return best_model

def predict(model, book):
    pred, err_rate = test(model, book, TEST_PATH)
    print("   🔸 Test error %.1f%% %s" % (err_rate * 100, symbol(err_rate)))
    return pred, err_rate

def write_out(pred, err_rate):
    test = pd.read_csv(TEST_PATH).copy()
    test["target"] = pred
    test.to_csv(OUTPUT_PATH, index=False)

    with open(LOG_PATH, 'a') as f:
        f.write(f"{err_rate}\n")

def main():
    start = time.time()

    print("🔹 Building dictionary...")
    book = dictionary()

    print("🔹 Training model...")
    model = train(book)

    print("🔹 Evaluating on test set...")
    pred, err = predict(model, book)

    print("🔹 Writing out...")
    write_out(pred, err)

    print(f"\n✅ Completed in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()
