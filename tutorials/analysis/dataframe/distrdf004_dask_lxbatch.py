# \file
# \ingroup tutorial_dataframe
#
# Configure a Dask connection to a HTCondor cluster hosted by the CERN batch
# service. To reproduce this tutorial, run the following steps:
#
# 1. Login to lxplus
# 2. Source an LCG release (minimum LCG104). See
#    https://lcgdocs.web.cern.ch/lcgdocs/lcgreleases/introduction/ for details
# 3. Install the `dask_lxplus` package, which provides the `CernCluster` class
#    needed to properly connect to the CERN condor pools. See
#    https://batchdocs.web.cern.ch/specialpayload/dask.html for instructions
# 4. Run this tutorial
#
# The tutorial defines resources that each job will request to the condor
# scheduler, then creates a Dask client that can be used by RDataFrame to
# distribute computations.
#
# \macro_code
#
# \date September 2023
# \author Vincenzo Eduardo Padulano CERN
from datetime import datetime
import socket
import time

from dask.distributed import Client
from dask_lxplus import CernCluster

import ROOT
RDataFrame = ROOT.RDF.Experimental.Distributed.Dask.RDataFrame


def create_connection() -> Client:
    """
    Creates a connection to HTCondor cluster offered by the CERN batch service.
    Returns a Dask client that RDataFrame will use to distribute computations.
    """
    # The resources described in the specified arguments to this class represent
    # the submission of a single job and will spawn a single Dask worker when
    # the condor scheduler launches the job. Specifically, this example has Dask
    # workers each with 1 core and 2 GB of memory.
    cluster = CernCluster(
        cores=1,
        memory='2000MB',
        disk='1GB',
        death_timeout='60',
        lcg=True,
        nanny=True,
        container_runtime='none',
        scheduler_options={
            'port': 8786,
            'host': socket.gethostname(),
        },
        job_extra={
            'MY.JobFlavour': '"espresso"',
        },
    )

    # The scale method allows to launch N jobs with the description above (thus
    # N Dask workers). Calling this method on the cluster object launches the
    # condor jobs (i.e. it is equivalent to `condor_submit myjob.sub`). In this
    # example, two jobs are requested so two Dask workers will be eventually
    # launched for a total of 2 cores.
    n_workers = 2
    cluster.scale(n_workers)

    # The Dask client can be created after the condor jobs have been submitted.
    # At this point, the jobs may or may not have actually started. Thus, it is
    # not guaranteed that the application already has the requested resources
    # available.
    client = Client(cluster)

    # It is possible to tell the Dask client to wait until the condor scheduler
    # has started the requested jobs and launched the Dask workers.
    # The client will wait until 'n_workers' workers have been launched. In this
    # example, the client waits for all the jobs requested to start before
    # continuing with the application.
    print(f"Waiting for {n_workers} workers to start.")
    start = time.time()
    client.wait_for_workers(n_workers)
    end = time.time()
    print(f"All workers are ready, took {round(end - start, 2)} seconds.")

    return client


def run_analysis(connection: Client) -> None:
    """
    Run a simple example with RDataFrame, using the previously created
    connection to the HTCondor cluster.
    """
    df = RDataFrame(10_000, daskclient=connection).Define(
        "x", "gRandom->Rndm() * 100")

    nentries = df.Count()
    meanv = df.Mean("x")
    maxv = df.Max("x")
    minv = df.Min("x")

    print(f"Dataset has {nentries.GetValue()} entries")
    print("Column x stats:")
    print(f"\tmean: {meanv.GetValue()}")
    print(f"\tmax: {maxv.GetValue()}")
    print(f"\tmin: {minv.GetValue()}")


if __name__ == "__main__":
    connection = create_connection()
    print(f"Starting the computations at {datetime.now()}")
    start = time.time()
    run_analysis(connection)
    end = time.time()
    print(f"Computations ended at {datetime.now()}, "
          f"took {round(end - start, 2)} seconds.")
