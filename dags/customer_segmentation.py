# Import necessary libraries and modules
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from src.lab import build_save_model, data_preprocessing, load_data, load_model_elbow

# Define default arguments for your DAG
default_args = {
    'owner': 'Pavanchytanya Dharmapuri',
    'start_date': datetime(2026, 10, 1),
    'retries': 1,  # Retry a failed task once
    'retry_delay': timedelta(minutes=2),
}

# Weekly run so the segments are refreshed as new orders come in
with DAG(
    'Customer_Segmentation_RFM',
    default_args=default_args,
    description='RFM customer segmentation with K-Means on the Online Retail dataset',
    schedule='@weekly',
    catchup=False,
    max_active_runs=1,  # runs share the files in working_data, so one at a time
    tags=['mlops', 'clustering'],
) as dag:

    # Task to load transactions and build the customer RFM table
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    # Task to split customers and scale the RFM features, depends on 'load_data_task'
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    # Task to pick k with the elbow method and save the model, depends on 'data_preprocessing_task'
    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "model.sav"],
    )

    # Task to load the model, label the segments and score new customers
    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=load_model_elbow,
        op_args=["model.sav", build_save_model_task.output],
    )

    # Set task dependencies
    load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task

# If this script is run directly, run the DAG once locally
if __name__ == "__main__":
    dag.test()
