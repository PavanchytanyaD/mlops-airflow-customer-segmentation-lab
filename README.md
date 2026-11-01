# Customer Segmentation with Airflow: MLOps Airflow Lab

This lab uses Apache Airflow to run a customer segmentation pipeline. A DAG of
four tasks loads a year of online store orders, builds recency, frequency and
monetary (RFM) features for each customer, picks the number of clusters with the
elbow method, trains a K-Means model and labels each segment. Airflow runs in
Docker Compose and the DAG is scheduled weekly.

> **Lab Assignment 3 submission.** Adapted from
> [Airflow_Labs/Lab_1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Airflow_Labs/Lab_1)
> in the [raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps) course
> repository. The structure is the same (one DAG, four PythonOperator tasks that
> pass data through XCom, K-Means with the elbow method, Airflow in Docker Compose),
> and the dataset, features, model selection and outputs are new. See
> [My Modifications](#my-modifications).

---

## Dataset

The [Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
from the UCI Machine Learning Repository. It is not used in any of the course labs.

- 541,909 transactions from a UK online gift store between December 2010 and
  December 2011, with invoice number, product, quantity, price, date, customer ID
  and country.
- Cleaning removes rows with no customer ID (135,080), cancelled invoices (invoice
  numbers starting with `C`), rows with zero or negative quantity or price, and
  exact duplicates. 392,692 transactions from 4,338 customers remain.
- Each customer gets three features:
  - **recency_days:** days since their last order, counted from the day after the
    last invoice in the data
  - **frequency:** number of separate orders
  - **monetary:** total amount spent (quantity x unit price)

The data is stored in `dags/data/online_retail.csv.gz` (converted from the original
Excel file). `dags/data/test.csv` has three new customers to segment.

License: CC BY 4.0. Chen, D. (2015). Online Retail. UCI Machine Learning Repository.
https://doi.org/10.24432/C5BW33

---

## My Modifications

The original lab clusters a credit card dataset on three columns, saves the last
model it fits (k=49), and prints the elbow value without using it. In this version:

- **Problem and dataset:** customer segmentation on 541,909 retail transactions
  instead of clustering pre-made credit card columns. The first task now turns raw
  orders into one row per customer.
- **Features:** RFM features built from the transactions. They are very skewed (the
  top customer spent £280,206, the median £669), so they are log-transformed before
  scaling.
- **Preprocessing:** the log + standard scaling pipeline is fitted on the train
  customers only and saved as `preprocessor.pkl`, so new customers in `test.csv`
  are scaled the same way before prediction.
- **Holdout check:** customers are split 80/20. The model is trained on 3,470
  customers and the silhouette score is compared on the 868 holdout customers.
- **Model selection:** K-Means is fitted for k = 2 to 10. The elbow of the SSE curve
  (found with `kneed`) picks k, the model is refit with that k and saved. If the curve
  has no clear elbow, the k with the best silhouette score is used.
- **Segment labels:** each cluster is described by its median recency, frequency and
  spend and its share of total revenue, ranked from most to least valuable, and given
  a plain name (Top customers, Loyal, At risk, New or occasional, Lost).
- **Outputs:** the last task writes `segment_report.json` (scores for every k, chosen
  k, silhouette scores, segment profiles, test predictions) and
  `customer_segments.csv` (the segment for every customer) to `working_data/`.
- **DAG:** scheduled `@weekly` instead of manual only, one retry per task,
  `max_active_runs=1` so two runs never write the same output files at once, and tags.
  The DAG file is named `customer_segmentation.py`, because a file named `airflow.py`
  in the dags folder hides the real `airflow` package when Python imports it.
- **Docker:**
  - The official Airflow 3.3.2 `docker-compose.yaml` is included, with example DAGs
    turned off and a `working_data` volume for the outputs.
  - A small `Dockerfile` extends the Airflow image with `kneed` instead of
    installing packages each time a container starts.
  - `setup.sh` creates the folders and `.env`, builds the image and initializes
    the database.
- **Tests:** pytest tests for the cleaning, RFM features, segment names, the full
  four-task pipeline on the real data, and the DAG structure. A GitHub Actions
  workflow runs them on every push.

---

## Results

The elbow of the SSE curve is at **k = 5**.

| k | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| SSE | 5,186 | 3,911 | 3,176 | **2,692** | 2,320 | 2,076 | 1,907 | 1,762 | 1,635 |
| Silhouette | 0.434 | 0.338 | 0.336 | **0.310** | 0.309 | 0.305 | 0.295 | 0.277 | 0.275 |

k = 2 has the highest silhouette score, but it only separates active customers from
inactive ones. k = 5 gives segments a store could act on, and its silhouette score
is about the same as k = 3 to 7.

The silhouette score is 0.310 on the train customers and 0.314 on the holdout
customers, so the segments hold up on customers the model has not seen.

| Segment | Customers | Days since last order (median) | Orders (median) | Spend (median) | Share of revenue |
|---|---|---|---|---|---|
| Top customers | 396 (9.1%) | 7 | 14 | £5,577 | 55.2% |
| Loyal | 953 (22.0%) | 16 | 5 | £1,699 | 21.9% |
| At risk | 1,013 (23.4%) | 85 | 3 | £965 | 16.0% |
| New or occasional | 851 (19.6%) | 25 | 1 | £342 | 3.6% |
| Lost | 1,125 (25.9%) | 217 | 1 | £234 | 3.3% |

- 9% of customers bring in 55% of the revenue.
- The At risk group has ordered a few times (median 3) but not for almost three
  months. It holds 16% of revenue, so it is the obvious target for a win-back email.
- About a quarter of customers (Lost) ordered once, a median of 7 months ago.

New customers from `test.csv`:

| Days since last order | Orders | Spend | Segment |
|---|---|---|---|
| 3 | 25 | £9,000 | Top customers |
| 40 | 4 | £1,200 | At risk |
| 200 | 1 | £150 | Lost |

![Customer_Segmentation_RFM DAG in the Airflow UI](assets/airflow_dag.jpg)

---

## Project Structure

```
.
├── docker-compose.yaml         # official Airflow 3.3.2 compose file, adapted
├── Dockerfile                  # Airflow image + kneed
├── setup.sh                    # folders, .env, build, database init
├── requirements.txt            # packages added to the Airflow image
├── requirements-dev.txt        # requirements + pandas and pytest for local tests
├── pytest.ini
├── dags/
│   ├── customer_segmentation.py   # the DAG
│   ├── src/lab.py                 # the four task functions
│   └── data/
│       ├── online_retail.csv.gz
│       └── test.csv
├── tests/                      # pytest tests
├── assets/                     # Airflow UI screenshot
└── .github/workflows/tests.yml # runs the tests on every push
```

`logs/`, `config/`, `plugins/` and `working_data/` are created by `setup.sh` and
filled by Airflow.

---

## How to Run

You need Docker Desktop with at least 4 GB of memory for Docker.

1. Set up (creates folders and `.env`, builds the image, initializes the database):

    ```sh
    ./setup.sh
    ```

2. Start Airflow:

    ```sh
    docker compose up -d
    ```

3. Open http://localhost:8080 and sign in with `airflow` / `airflow`.

4. Find `Customer_Segmentation_RFM`, turn it on and click **Trigger**. A run takes
   about 20 seconds.

5. The results are in `working_data/`: `segment_report.json`,
   `customer_segments.csv`, `model.sav` and `preprocessor.pkl`.

6. Stop Airflow:

    ```sh
    docker compose down
    ```

### Running the tests

```sh
pip install -r requirements-dev.txt
pytest
```

The DAG structure tests need Airflow, so they are skipped outside the container. To
run every test inside the Airflow image:

```sh
docker compose run --rm -v "$PWD":/repo --entrypoint bash airflow-cli \
  -c "pip install -q pytest && cd /repo && python -m pytest -p no:cacheprovider"
```
