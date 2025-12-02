import numpy as np
import pandas as pd
import time

from gensim.models import KeyedVectors
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

TRAIN_PATH, DEV_PATH, TEST_PATH, EMBS_PATH = "train.csv", "dev.csv", "test.csv", "embs_train.kv"
OUTPUT_PATH, LOG_PATH = "test.predicted.csv", "log.txt"

EMB_DIM = 300


# -------------------- Common --------------------
def load(path):
    df = pd.read_csv(path)
    ids = df['id']
    x = df['sentence'].astype(str).str.split().tolist()
    y = df['target'] if 'target' in df.columns else None
    if y is not None:
        y = y.map(lambda v: 1 if v == '+' else -1)
    return ids, x, y

def load_embed(sentences, wv):
    x_emb = np.zeros((len(sentences), EMB_DIM), dtype=np.float32)
    for i, toks in enumerate(sentences):
        vecs = [wv[w] for w in toks if w in wv]
        if vecs:
            x_emb[i] = np.mean(vecs, axis=0)
    return x_emb

def write(y_pred, err, name):
    df = pd.read_csv(TEST_PATH).copy()
    df['target'] = y_pred
    df.to_csv(OUTPUT_PATH, index=False)
    try:
        vals = [*map(float, open(LOG_PATH).read().split())]
    except (FileNotFoundError, ValueError):
        vals = []
    prev_last = vals[-1] if vals else None
    prev_best = min(vals) if vals else None
    if err is not None:
        with open(LOG_PATH, 'a') as f:
            f.write(f"{err:.6f}\n")
    def s(x, y):
        if y is None or x is None or abs(x - y) < 1e-6:
            return '='
        return '+' if x < y else '-'
    shown = f"{err:.3f}" if err is not None else "?"
    print(f"  🔹 {name} Err Rate {shown} L{s(err, prev_last)}{s(err, prev_best)}B")


# -------------------- Standard Perceptron --------------------
def train_percept(x_train, y_train, x_dev, y_dev):
    clf = SGDClassifier(loss='perceptron', penalty=None, learning_rate='constant', eta0=1.0, max_iter=10, random_state=0)
    clf.fit(x_train, y_train)
    dev_error = np.mean(clf.predict(x_dev) != y_dev)
    print(f"        🔸 Best Dev Err: {dev_error:.3f}")
    return clf, dev_error

def predict_percept(model, x_test):
    print("    🔹 Predicting...\n")
    y_pred_int = model.predict(x_test)
    return ['+' if v == 1 else '-' for v in y_pred_int]


# -------------------- Averaged Perceptron --------------------
def train_avg_percept(x_train, y_train, x_dev, y_dev):
    clf = SGDClassifier(loss='perceptron', penalty=None, learning_rate='constant', eta0=1.0, average=True, max_iter=10, random_state=0)
    clf.fit(x_train, y_train)
    dev_error = np.mean(clf.predict(x_dev) != y_dev)
    print(f"        🔸 Best Dev Err: {dev_error:.3f}")
    return clf, dev_error

def predict_avg_percept(model, x_test):
    print("    🔹 Predicting...\n")
    y_pred_int = model.predict(x_test)
    return ['+' if v == 1 else '-' for v in y_pred_int]


# -------------------- Averaged One-count Perceptron --------------------
def train_avg_oc_percept(r_x_train, y_train, r_x_dev, y_dev, r_x_test, wv):
    train_texts = [" ".join(toks) for toks in r_x_train]
    cv = CountVectorizer(min_df=2)
    cv.fit(train_texts)
    all_tokens = set(w for toks in r_x_train for w in toks)
    kept_tokens = set(cv.vocabulary_.keys())
    prune_tokens = all_tokens - kept_tokens
    print(f"    🔸 Dropped {len(prune_tokens)}, Kept {len(kept_tokens)}")
    def load_embed_pruned(sentences, wv_local, prune_set):
        emb_dim = wv_local.vector_size
        X = np.zeros((len(sentences), emb_dim), dtype=np.float32)
        for i, toks in enumerate(sentences):
            vecs = [wv_local[w] for w in toks if w not in prune_set and w in wv_local]
            X[i] = np.mean(vecs, axis=0) if vecs else np.zeros(emb_dim, dtype=np.float32)
        return X

    x_train_emb = load_embed_pruned(r_x_train, wv, prune_tokens)
    x_dev_emb   = load_embed_pruned(r_x_dev,   wv, prune_tokens)
    x_test_emb  = load_embed_pruned(r_x_test,  wv, prune_tokens)

    clf = SGDClassifier(loss='perceptron', penalty=None,
                        learning_rate='constant', eta0=1.0,
                        average=True, max_iter=10, random_state=0)
    clf.fit(x_train_emb, y_train)
    dev_error = np.mean(clf.predict(x_dev_emb) != y_dev)
    print(f"        🔸 Best Dev Err: {dev_error:.3f}")

    return clf, dev_error, x_test_emb

def predict_avg_oc_percept(model, x_test, keep_mask=None):
    print("    🔹 Predicting...\n")
    if keep_mask is not None:
        x_test = x_test[:, keep_mask]
    y_pred_int = model.predict(x_test)
    return ['+' if v == 1 else '-' for v in y_pred_int]


# -------------------- KNN --------------------
def train_knn(x_train, y_train, x_dev, y_dev, ks=range(1, 100, 2), DEBUG=False):
    best_clf, best_k, best_err = None, None, 1.0
    for k in ks:
        clf = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
        clf.fit(x_train, y_train)
        dev_err = float(np.mean(clf.predict(x_dev) != y_dev))
        if DEBUG:
            print(f"🔸 K = {k:2d}, Dev Err = {dev_err:.3f}")
        if dev_err < best_err:
            best_clf, best_k, best_err = clf, k, dev_err
    print(f"    🔸 Best K = {best_k}")
    print(f"        🔸 Dev Err = {best_err:.3f}")
    return best_k, best_err, best_clf

def predict_knn(model, x_test):
    print("    🔹 Predicting...\n")
    y_pred_int = model.predict(x_test)
    return ['+' if int(v) == 1 else '-' for v in y_pred_int]


# -------------------- One-hot KNN --------------------
def train_oh_knn(train_tokens, y_train, dev_tokens, y_dev, ks=range(1, 100, 2), DEBUG=False):
    train_texts = [" ".join(toks) for toks in train_tokens]
    dev_texts = [" ".join(toks) for toks in dev_tokens]

    best_clf, best_k, best_err = None, None, 1.0
    for k in ks:
        clf = make_pipeline(
            CountVectorizer(binary=True),
            KNeighborsClassifier(n_neighbors=k, metric='manhattan')
        )
        clf.fit(train_texts, y_train)
        dev_err = float(np.mean(clf.predict(dev_texts) != y_dev))
        if DEBUG:
            print(f"🔸 K = {k:2d}, Dev Err = {dev_err:.3f}")
        if dev_err < best_err:
            best_clf, best_k, best_err = clf, k, dev_err

    print(f"    🔸 Best K = {best_k}")
    print(f"        🔸 Dev Err = {best_err:.3f}")
    return best_clf, best_err, best_k

def predict_oh_knn(clf, test_tokens):
    print("    🔹 Predicting...\n")
    test_texts = [" ".join(toks) for toks in test_tokens]
    y_pred_int = clf.predict(test_texts)
    return ['+' if v == 1 else '-' for v in y_pred_int]


# -------------------- SVM --------------------
def train_svm(x_train, y_train, x_dev, y_dev):
    clf = LinearSVC(max_iter=1000)
    clf.fit(x_train, y_train)
    dev_error = np.mean(clf.predict(x_dev) != y_dev)
    print(f"        🔸 Dev Err = {dev_error:.3f}")
    return clf, dev_error

def predict_svm(model, x_test):
    print("    🔹 Predicting...\n")
    y_pred_int = model.predict(x_test)
    return ['+' if v == 1 else '-' for v in y_pred_int]


# -------------------- Main --------------------
def main():
    print("🔹 Loading data...")
    _, r_x_train, r_y_train = load(TRAIN_PATH)
    _, r_x_dev,   r_y_dev   = load(DEV_PATH)
    _, r_x_test,  _         = load(TEST_PATH)

    print("🔹 Loading embeddings...")
    wv = KeyedVectors.load(EMBS_PATH)

    x_train_emb = load_embed(r_x_train, wv)
    x_dev_emb   = load_embed(r_x_dev,   wv)
    x_test_emb  = load_embed(r_x_test,  wv)

    y_train = np.asarray(r_y_train, dtype=int)
    y_dev   = np.asarray(r_y_dev,   dtype=int)

    models = []

    # ---- Standard Perceptron ----
    print("🔹 Training Perceptron...")
    start = time.time()
    model, dev_error = train_percept(x_train_emb, y_train, x_dev_emb, y_dev)
    end = time.time()
    print(f"    🔸 Training time: {end - start:.2f}s")
    y_pred = predict_percept(model, x_test_emb)
    models.append({"n": "Standard Perceptron", "e": dev_error, "p": y_pred})

    # ---- Averaged Perceptron ----
    print("🔹 Training Averaged Perceptron...")
    start = time.time()
    model, dev_error = train_avg_percept(x_train_emb, y_train, x_dev_emb, y_dev)
    end = time.time()
    print(f"    🔸 Training time: {end - start:.2f}s")
    y_pred = predict_avg_percept(model, x_test_emb)
    models.append({"n": "Averaged Perceptron", "e": dev_error, "p": y_pred})

    # ---- One-Count Averaged Perceptron ----
    print("🔹 Training Averaged One-Count Perceptron...")
    start = time.time()
    model, dev_error, x_test_emb_pruned = train_avg_oc_percept(
        r_x_train, y_train, r_x_dev, y_dev, r_x_test, wv
    )
    end = time.time()
    print(f"    🔸 Training time: {end - start:.2f}s")
    y_pred = predict_avg_oc_percept(model, x_test_emb_pruned)
    models.append({"n": "Averaged One-count Perceptron", "e": dev_error, "p": y_pred})

    # ---- KNN ----
    print("🔹 Training KNN...")
    start = time.time()
    best_k, dev_error, knn_model = train_knn(x_train_emb, y_train, x_dev_emb, y_dev, ks=range(1, 10, 2))
    end = time.time()
    print(f"    🔸 Training time: {end - start:.2f}s")
    y_pred = predict_knn(knn_model, x_test_emb)
    models.append({"n": f"KNN (k={best_k})", "e": dev_error, "p": y_pred})

    # ---- One-hot KNN ----
    print("🔹 Training One-Hot KNN...")
    start = time.time()
    clf, dev_error, best_k = train_oh_knn(r_x_train, y_train, r_x_dev, y_dev, ks=range(1, 10, 2))
    end = time.time()
    print(f"    🔸 Training time: {end - start:.2f}s")
    y_pred = predict_oh_knn(clf, r_x_test)
    models.append({"n": f"One-hot KNN (k={best_k})", "e": dev_error, "p": y_pred})

    # ---- SVM ----
    print("🔹 Training SVM...")
    start = time.time()
    model, dev_error = train_svm(x_train_emb, y_train, x_dev_emb, y_dev)
    end = time.time()
    print(f"    🔸 Training time: {end - start:.2f}s")
    y_pred = predict_svm(model, x_test_emb)
    models.append({"n": "SVM", "e": dev_error, "p": y_pred})



    name, dev_err, y_pred = (lambda m: (m["n"], m["e"], m["p"]))(min(models, key=lambda m: m["e"]))
    print("🔹 Writing best output...")
    write(y_pred, dev_err, name)


if __name__ == "__main__":
    main()
