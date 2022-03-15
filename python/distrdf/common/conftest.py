"""
pytest automatically loads fixtures written in a conftest.py module in the same
folder as other tests. This module creates both a Dask cluster as well as a
Spark cluster, on the local machine.
"""
from functools import partial

import pytest

import pyspark
from dask.distributed import Client, LocalCluster

import ROOT


@pytest.fixture(scope="session")
def connection():
    """
    Creates a mock Dask cluster as well as a mock Spark cluster. Returns a tuple
    with both connection objects.

    Note that the name of this fixture will be used to inject the result into an
    equally named variable that should be used as input argument to all tests
    that need it. Example:

    ```
    def test_my_feature(connection):
        # The tests in this folder need both dask and spark connections
        daskclient, sparkcontext = connection
        df1 = RDataFrame(10, daskclient=daskclient)
        df2 = RDataFrame(10, sparkcontext=sparkcontext)
    ```
    """
    daskconn = Client(LocalCluster(n_workers=2, threads_per_worker=1, processes=True, memory_limit="2GiB"))

    conf = {"spark.master": "local[2]", "spark.driver.memory": "4g", "spark.app.name": "roottest-distrdf-common"}
    sparkconf = pyspark.SparkConf().setAll(conf.items())
    sparkconn = pyspark.SparkContext(conf=sparkconf)

    yield daskconn, sparkconn

    daskconn.shutdown()
    daskconn.close()
    sparkconn.stop()


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
