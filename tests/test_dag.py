# Checks the DAG file loads and wires the tasks in the right order.
# Needs Airflow installed, so it is skipped elsewhere (it runs inside the Airflow container).
import os

import pytest

pytest.importorskip('airflow')
from airflow.models import DagBag  # noqa: E402

DAGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'dags')


@pytest.fixture(scope='module')
def dag():
    bag = DagBag(dag_folder=DAGS_DIR)
    assert bag.import_errors == {}
    return bag.get_dag('Customer_Segmentation_RFM')


def test_tasks_in_order(dag):
    order = ['load_data_task', 'data_preprocessing_task', 'build_save_model_task', 'load_model_task']
    assert sorted(dag.task_ids) == sorted(order)
    for upstream, downstream in zip(order, order[1:]):
        assert dag.get_task(upstream).downstream_task_ids == {downstream}


def test_dag_settings(dag):
    assert dag.max_active_runs == 1
    assert dag.catchup is False
