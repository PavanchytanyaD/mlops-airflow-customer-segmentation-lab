import json
import os

import pandas as pd
import pytest

from src import lab


def transactions(rows):
    cols = ['InvoiceNo', 'StockCode', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID']
    df = pd.DataFrame(rows, columns=cols)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df


def test_clean_transactions_drops_bad_rows():
    df = transactions([
        ['1001', 'A', 2, '2011-01-01', 3.0, 1.0],      # kept
        ['1001', 'A', 2, '2011-01-01', 3.0, 1.0],      # duplicate
        ['C1002', 'A', -2, '2011-01-02', 3.0, 1.0],    # cancellation
        ['1003', 'B', 1, '2011-01-03', 5.0, None],     # no customer
        ['1004', 'B', 0, '2011-01-04', 5.0, 2.0],      # zero quantity
        ['1005', 'B', 1, '2011-01-05', 0.0, 2.0],      # free item
    ])
    clean = lab.clean_transactions(df)
    assert clean['InvoiceNo'].tolist() == ['1001']


def test_build_rfm():
    df = transactions([
        ['1', 'A', 2, '2011-01-01', 10.0, 7.0],
        ['2', 'A', 1, '2011-01-08', 5.0, 7.0],
        ['2', 'B', 3, '2011-01-08', 1.0, 7.0],   # same invoice, counts once
        ['3', 'A', 1, '2011-01-05', 4.0, 9.0],
    ])
    rfm = lab.build_rfm(df, pd.Timestamp('2011-01-10'))
    assert rfm.loc[7].tolist() == [2, 2, 28.0]   # 2 days, 2 orders, 20 + 5 + 3
    assert rfm.loc[9].tolist() == [5, 1, 4.0]


def test_b64_round_trip():
    df = pd.DataFrame({'a': [1, 2]})
    assert lab.from_b64(lab.to_b64(df)).equals(df)


@pytest.mark.parametrize('recency, monetary, name', [
    (5, 9000, 'Top customers'),
    (10, 2000, 'Loyal'),
    (10, 100, 'New or occasional'),
    (200, 2000, 'At risk'),
    (200, 100, 'Lost'),
])
def test_segment_name(recency, monetary, name):
    overall = pd.Series({'recency_days': 50, 'monetary': 700})
    profile = pd.Series({'recency_days': recency, 'monetary': monetary})
    assert lab.segment_name(profile, overall, top_spend=5000) == name


@pytest.fixture(scope='module')
def pipeline_run(tmp_path_factory):
    # Runs the four DAG tasks in order, the way Airflow chains them
    out = tmp_path_factory.mktemp('working_data')
    old = lab.OUTPUT_DIR
    lab.OUTPUT_DIR = str(out)
    try:
        data = lab.load_data()
        prepared = lab.data_preprocessing(data)
        scores = lab.build_save_model(prepared, 'model.sav')
        predictions = lab.load_model_elbow('model.sav', scores)
    finally:
        lab.OUTPUT_DIR = old
    return out, data, scores, predictions


def test_rfm_table(pipeline_run):
    _, data, _, _ = pipeline_run
    rfm = lab.from_b64(data)
    assert len(rfm) == 4338
    assert list(rfm.columns) == lab.RFM_COLUMNS
    assert (rfm['frequency'] >= 1).all() and (rfm['monetary'] > 0).all()
    assert rfm['recency_days'].min() >= 0


def test_scores_are_json_safe(pipeline_run):
    _, _, scores, _ = pipeline_run
    json.dumps(scores)
    assert scores['k_values'] == list(lab.K_VALUES)
    assert scores['best_k'] in scores['k_values']
    # SSE always goes down as k goes up
    assert all(a > b for a, b in zip(scores['sse'], scores['sse'][1:]))


def test_outputs_written(pipeline_run):
    out, _, _, _ = pipeline_run
    for name in ['preprocessor.pkl', 'model.sav', 'customers_rfm.csv',
                 'customer_segments.csv', 'segment_report.json']:
        assert os.path.exists(os.path.join(out, name)), name


def test_report(pipeline_run):
    out, _, scores, _ = pipeline_run
    report = json.load(open(os.path.join(out, 'segment_report.json')))
    segments = report['segments']
    assert report['best_k'] == scores['best_k'] == len(segments)
    assert sum(s['customers'] for s in segments) == 4338
    assert sum(s['revenue_share'] for s in segments) == pytest.approx(1, abs=0.01)
    # holdout customers should cluster about as well as the train customers
    assert report['silhouette_holdout'] > 0.2
    assert abs(report['silhouette_holdout'] - report['silhouette_train']) < 0.05
    # the best segment comes first
    assert segments[0]['monetary'] == max(s['monetary'] for s in segments)


def test_new_customers(pipeline_run):
    _, _, _, predictions = pipeline_run
    assert predictions[0] == 'Top customers'   # bought 3 days ago, 25 orders, 9,000
    assert predictions[-1] == 'Lost'           # last order 200 days ago, 1 order, 150
