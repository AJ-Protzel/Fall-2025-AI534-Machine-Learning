import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_log_error

TRAIN_PATH, TEST_PATH, OUTPUT_PATH = "train.csv", "test.csv", "test.predicted.csv"
MY_TRAIN_PATH, MY_TEST_PATH, LOG_PATH = "my_train.csv", "my_dev.csv", "log.txt"

def load(path, enc=None):
    # Read CSV and ex_trainact IDs
    df = pd.read_csv(path)
    ids = df['Id'].to_numpy()
    
    # Drop Id and SalePrice to get feature matrix
    x = df.drop(['Id','SalePrice'], axis=1, errors='ignore')
    
    # If no encoder provided, fit OneHotEncoder on categorical columns
    if enc is None:
        cat = x.select_dtypes(include=['object','string']).columns.tolist()
        x[cat] = x[cat].astype(str).fillna("NA") if cat else x
        ohe = OneHotEncoder(handle_unknown='ignore')
        if cat: ohe.fit(x[cat])
        enc = {'ohe': ohe, 'categorical_cols': cat, 'numeric_cols': [c for c in x.columns if c not in cat]}
    else:
        # Use existing encoder for transformation
        cat = enc['categorical_cols']
        if cat: x[cat] = x[cat].astype(str).fillna("NA")
    
    # Transform categorical and numeric data
    cat_data = enc['ohe'].transform(x[cat]).toarray() if cat else np.empty((len(x),0))
    num_data = x[enc['numeric_cols']].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(float) if enc['numeric_cols'] else np.empty((len(x),0))
    
    # Combine numeric and categorical features
    X = np.hstack([num_data, cat_data]) if cat_data.size else num_data
    
    # Ex_trainact target if present
    y = df['SalePrice'].astype(float).to_numpy() if 'SalePrice' in df.columns else None
    return ids, X, y, enc

def tune_alpha(x_train, y_train, x_my_test, y_my_test):
    # Search best alpha using logspace and refinement
    best_a, best_e = None, float('inf')
    for a in np.logspace(-4, 4, 40):
        e = _eval(x_train, y_train, x_my_test, y_my_test, a)
        if e < best_e: best_e, best_a = e, a
    for _ in range(2):
        for a in np.geomspace(max(1e-6, best_a/3), best_a*3, 24):
            e = _eval(x_train, y_train, x_my_test, y_my_test, a)
            if e < best_e: best_e, best_a = e, a
    print(f"    🔸 Best alpha {best_a:.6f}")
    return best_a

def _eval(x_train, y_train, x_my_test, y_my_test, a):
    # Train Ridge with log target and compute RMSLE
    m = Ridge(alpha=a, max_iter=10000).fit(x_train, np.log1p(y_train))
    yp = np.maximum(np.expm1(m.predict(x_my_test)), 0)
    return np.sqrt(mean_squared_log_error(np.maximum(y_my_test, 0), yp))

def train(x, y, a):
    # Train Ridge regression on log-transformed target
    return Ridge(alpha=a, max_iter=10000).fit(x, np.log1p(y))

def predict(m, x, y=None):
    # Predict and optionally compute RMSLE
    yp = np.expm1(m.predict(x))
    return np.sqrt(mean_squared_log_error(y, yp)) if y is not None else yp

def poly(X, enc, fit=True, d=2):
    # Get number of numeric columns
    n = len(enc['numeric_cols'])
    num = X[:, :n]

    if fit:
        # Fit polynomial transformer on numeric features
        poly = PolynomialFeatures(d, include_bias=False)
        poly_num = poly.fit_transform(num)

        # Compute mean and std for normalization
        mean, std = poly_num.mean(0), poly_num.std(0)
        std[std == 0] = 1  # Avoid division by zero

        # Store transformer and normalization stats in encoder dict
        enc.update({'poly': poly, 'poly_mean': mean, 'poly_std': std})
    else:
        # Apply previously fitted polynomial transform
        poly_num = enc['poly'].transform(num)
        mean, std = enc['poly_mean'], enc['poly_std']

    # Normalize polynomial features
    poly_num = ((poly_num - mean) / std).astype(np.float32)

    # Combine normalized polynomial features with categorical part
    return np.hstack([poly_num, X[:, n:]])

def write(ids, y, e):
    # Save predictions and log RMSLE
    pd.DataFrame({'Id': ids, 'SalePrice': y}).to_csv(OUTPUT_PATH, index=False)

    with open(LOG_PATH, 'a') as f: f.write(f"{e}\n")

    vals = list(map(float, open(LOG_PATH).read().split()))
    last, best = vals[-1], min(vals)
    s = lambda x, y: '=' if abs(x - y) < 1e-6 else ('+' if x < y else '-')
    print(f"    🔸 RMSLE {e:.6f} L{s(e,last)}{s(e,best)}B")

def main():
    print("🔹 Loading data...")
    _, x_my_train, y_my_train, my_enc = load(MY_TRAIN_PATH)
    _, x_my_test, y_my_test, _ = load(MY_TEST_PATH, my_enc)
    _, x_train, y_train, enc = load(TRAIN_PATH)
    id_test, x_test, _, _ = load(TEST_PATH, enc)

    print("🔹 Applying polynomial expansion...")
    x_my_train = poly(x_my_train, my_enc, fit=True)
    x_my_test = poly(x_my_test, my_enc, fit=False)
    x_train = poly(x_train, enc, fit=True)
    x_test = poly(x_test, enc, fit=False)

    print("🔹 Tuning alpha ...")
    a = tune_alpha(x_my_train, y_my_train, x_my_test, y_my_test)

    print("🔹 Training model...")
    my_model = train(x_my_train, y_my_train, a)
    model = train(x_train, y_train, a)

    print("🔹 Predicting...")
    err = predict(my_model, x_my_test, y_my_test)
    y_pred = predict(model, x_test)

    print("🔹 Writing output...")
    write(id_test, y_pred, err)

if __name__=="__main__":main()