"""
pytest automatically loads fixtures written in a conftest.py module in the same
folder as other tests. This module creates a Spark cluster on the local machine.
"""
from functools import partial

import pytest

import pyspark

import ROOT


@pytest.fixture(scope="session")
def connection():
    """
    Creates a mock Spark cluster. Returns the corresponding connection object.

    Note that the name of this fixture will be used to inject the result into an
    equally named variable that should be used as input argument to all tests
    that need it. Example:

    ```
    def test_my_feature(connection):
        df = RDataFrame(10, sparkcontext=connection)
    ```
    """
    conf = {"spark.master": "local[2]", "spark.driver.memory": "4g", "spark.app.name": "roottest-distrdf-spark"}
    sparkconf = pyspark.SparkConf().setAll(conf.items())
    connection = pyspark.SparkContext(conf=sparkconf)
    yield connection
    connection.stop()


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
