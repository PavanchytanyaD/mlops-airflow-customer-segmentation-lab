import base64
import json
import os
import pickle

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

DAGS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(DAGS_DIR, "data")
# Outputs go to working_data/ next to the dags folder (mounted as a volume in Docker)
OUTPUT_DIR = os.environ.get("SEGMENTATION_OUTPUT_DIR",
                            os.path.join(os.path.dirname(DAGS_DIR), "working_data"))

RFM_COLUMNS = ["recency_days", "frequency", "monetary"]
K_VALUES = range(2, 11)
SEED = 42


def to_b64(obj):
    # XCom only stores JSON, so pickled objects are passed as base64 strings
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def from_b64(data_b64):
    return pickle.loads(base64.b64decode(data_b64))


def clean_transactions(df):
    """Keep real sales: known customer, not a cancellation, positive quantity and price."""
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    return df.drop_duplicates()


def build_rfm(df, snapshot_date):
    """One row per customer: days since last purchase, number of orders, total spend."""
    df = df.assign(revenue=df["Quantity"] * df["UnitPrice"])
    rfm = df.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        frequency=("InvoiceNo", "nunique"),
        monetary=("revenue", "sum"),
    )
    rfm["recency_days"] = (snapshot_date - rfm["last_purchase"]).dt.days
    rfm.index = rfm.index.astype(int)
    return rfm[RFM_COLUMNS]


def load_data():
    """
    Loads the Online Retail transactions, removes cancellations and bad rows,
    and returns the customer-level RFM table as a base64 string.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "online_retail.csv.gz"),
                     dtype={"InvoiceNo": str, "StockCode": str},
                     parse_dates=["InvoiceDate"])
    raw_rows = len(df)
    df = clean_transactions(df)
    # Measure recency from the day after the last invoice in the data
    snapshot_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = build_rfm(df, snapshot_date)
    print(f"{raw_rows} transactions -> {len(df)} after cleaning -> {len(rfm)} customers")
    return to_b64(rfm)


def data_preprocessing(data_b64):
    """
    Splits customers 80/20 into train and holdout, fits a log + standard scaling
    pipeline on the train customers only and saves it, and returns both splits.
    """
    rfm = from_b64(data_b64)
    train_ids, holdout_ids = train_test_split(rfm.index, test_size=0.2, random_state=SEED)

    # RFM values are heavily skewed (a few customers spend 100x the median),
    # so log1p comes before scaling
    preprocessor = make_pipeline(FunctionTransformer(np.log1p), StandardScaler())
    preprocessor.fit(rfm.loc[train_ids, RFM_COLUMNS])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "preprocessor.pkl"), "wb") as f:
        pickle.dump(preprocessor, f)
    rfm.assign(split=np.where(rfm.index.isin(train_ids), "train", "holdout")) \
       .to_csv(os.path.join(OUTPUT_DIR, "customers_rfm.csv"), index_label="CustomerID")

    return to_b64({
        "train": preprocessor.transform(rfm.loc[train_ids, RFM_COLUMNS]),
        "holdout": preprocessor.transform(rfm.loc[holdout_ids, RFM_COLUMNS]),
    })


def build_save_model(data_b64, filename):
    """
    Fits K-Means for k = 2..10 on the train customers, picks k with the elbow
    method, refits the model with that k and saves it. Returns the scores per k.
    """
    X_train = from_b64(data_b64)["train"]

    sse, silhouette = [], []
    for k in K_VALUES:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_train)
        sse.append(float(kmeans.inertia_))
        silhouette.append(float(silhouette_score(X_train, kmeans.labels_)))

    knee = KneeLocator(list(K_VALUES), sse, curve="convex", direction="decreasing").elbow
    # Fall back to the best silhouette score if the SSE curve has no clear elbow
    best_k = int(knee) if knee else list(K_VALUES)[int(np.argmax(silhouette))]
    print(f"Elbow at k={knee}, using k={best_k}")

    model = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X_train)
    with open(os.path.join(OUTPUT_DIR, filename), "wb") as f:
        pickle.dump(model, f)

    return {"k_values": list(K_VALUES), "sse": sse, "silhouette": silhouette, "best_k": best_k}


def segment_name(profile, overall, top_spend):
    """Plain-language name for a segment from its median recency and spend."""
    if profile["monetary"] >= top_spend:
        return "Top customers"
    recent = profile["recency_days"] <= overall["recency_days"]
    valuable = profile["monetary"] >= overall["monetary"]
    if recent and valuable:
        return "Loyal"
    if recent:
        return "New or occasional"
    if valuable:
        return "At risk"
    return "Lost"


def load_model_elbow(filename, scores):
    """
    Loads the saved model and preprocessor, assigns every customer to a segment,
    checks the segments on the holdout customers, describes each segment and
    predicts the segment for the new customers in test.csv.
    """
    with open(os.path.join(OUTPUT_DIR, filename), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "preprocessor.pkl"), "rb") as f:
        preprocessor = pickle.load(f)
    customers = pd.read_csv(os.path.join(OUTPUT_DIR, "customers_rfm.csv"), index_col="CustomerID")

    X = preprocessor.transform(customers[RFM_COLUMNS])
    customers["cluster"] = model.predict(X)
    is_train = (customers["split"] == "train").to_numpy()
    train_sil = silhouette_score(X[is_train], customers["cluster"][is_train])
    holdout_sil = silhouette_score(X[~is_train], customers["cluster"][~is_train])

    # Describe each cluster with medians, then rank clusters from most to least valuable
    overall = customers[RFM_COLUMNS].median()
    top_spend = customers["monetary"].quantile(0.9)
    profiles = customers.groupby("cluster").agg(
        customers=("monetary", "size"),
        recency_days=("recency_days", "median"),
        frequency=("frequency", "median"),
        monetary=("monetary", "median"),
        revenue_share=("monetary", "sum"),
    )
    profiles["revenue_share"] = profiles["revenue_share"] / customers["monetary"].sum()
    value_rank = (profiles["recency_days"].rank()
                  + profiles["frequency"].rank(ascending=False)
                  + profiles["monetary"].rank(ascending=False))
    profiles = profiles.loc[value_rank.sort_values().index]
    profiles["segment"] = [segment_name(p, overall, top_spend) for _, p in profiles.iterrows()]
    names = profiles["segment"].to_dict()
    customers["segment"] = customers["cluster"].map(names)
    customers.to_csv(os.path.join(OUTPUT_DIR, "customer_segments.csv"))

    # New customers are scaled with the saved preprocessor before predicting
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    test["segment"] = pd.Series(model.predict(preprocessor.transform(test[RFM_COLUMNS]))).map(names)

    report = {
        "best_k": int(model.n_clusters),
        "scores_by_k": scores,
        "silhouette_train": round(float(train_sil), 3),
        "silhouette_holdout": round(float(holdout_sil), 3),
        "segments": profiles.reset_index().round(3).to_dict(orient="records"),
        "test_predictions": test.to_dict(orient="records"),
    }
    with open(os.path.join(OUTPUT_DIR, "segment_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"k={model.n_clusters}, silhouette train {train_sil:.3f}, holdout {holdout_sil:.3f}")
    print(profiles.to_string())
    print(test.to_string(index=False))
    return test["segment"].tolist()
