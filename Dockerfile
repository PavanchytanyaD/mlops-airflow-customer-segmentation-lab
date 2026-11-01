# Official Airflow image plus the packages the DAG needs
FROM apache/airflow:3.3.2

COPY requirements.txt /requirements.txt
# Pinning apache-airflow keeps pip from upgrading Airflow itself
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
