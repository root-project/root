"""
pytest automatically loads fixtures written in a conftest.py module in the same
folder as other tests. This module creates a Dask cluster on the local machine.
"""
from functools import partial

import pytest
import os

from dask.distributed import Client, LocalCluster
import pyspark
import ROOT


def create_dask_connection():
    connection = Client(LocalCluster(
        n_workers=2, threads_per_worker=1, processes=True, memory_limit="2GiB"))
    return connection


def cleanup_dask_connection(connection):
    connection.close()


def create_spark_connection():
    conf = {"spark.master": "local[2]", "spark.driver.memory": "4g",
            "spark.app.name": "roottest-distrdf-spark"}
    sparkconf = pyspark.SparkConf().setAll(conf.items())
    connection = pyspark.SparkContext(conf=sparkconf)
    return connection


def cleanup_spark_connection(connection):
    connection.stop()


CONNECTION_FNS = {
    "dask": (create_dask_connection, cleanup_dask_connection),
    "spark": (create_spark_connection, cleanup_spark_connection)
}


@pytest.fixture(scope="session", params=os.environ["DISTRDF_BACKENDS_IN_USE"].split(","))
def payload(request):
    """
    Creates a mock cluster.
    Returns:
        A tuple with the corresponding connection object and a string
        representing the backend type.

    Note that the name of this fixture will be used to inject the result into an
    equally named variable that should be used as input argument to all tests
    that need it. Example:

    ```
    def test_my_feature(payload):
        connection, backend = payload
        df = RDataFrame(10, executor=connection)
    ```

    The environment variable "DISTRDF_BACKENDS_IN_USE" is set at configuration
    time by CMake and injects one or more backend types for use as parameters
    to create the corresponding backend connections.
    Currently the value can be "dask","spark" or both.
    """
    try:
        create, cleanup = CONNECTION_FNS[request.param]
    except AttributeError as e:
        raise RuntimeError(f"Backend '{request.param}' not supported!") from e

    connection = create()
    yield (connection, request.param)
    cleanup(connection)


@pytest.fixture(scope="session", autouse=True)
def session_setup_teardown(request):
    """
    Session setup / teardown. Includes:
    - Disable graphics in tests
    """
    old_batch = ROOT.gROOT.IsBatch()
    ROOT.gROOT.SetBatch(True)

    def restore_setbatch(old_batch):
        ROOT.gROOT.SetBatch(old_batch)

    request.addfinalizer(partial(restore_setbatch, old_batch))
