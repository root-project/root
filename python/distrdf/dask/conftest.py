"""
pytest automatically loads fixtures written in a conftest.py module in the same
folder as other tests. This module creates a Dask cluster on the local machine.
"""
from functools import partial

import pytest

from dask.distributed import Client, LocalCluster

import ROOT


@pytest.fixture(scope="session")
def connection():
    """
    Creates a mock Dask cluster. Returns the corresponding connection object.

    Note that the name of this fixture will be used to inject the result into an
    equally named variable that should be used as input argument to all tests
    that need it. Example:

    ```
    def test_my_feature(connection):
        df = RDataFrame(10, daskclient=connection)
    ```
    """
    connection = Client(LocalCluster(n_workers=2, threads_per_worker=1, processes=True, memory_limit="2GiB"))
    yield connection
    connection.shutdown()
    connection.close()


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
