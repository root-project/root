## \file
## \ingroup tutorial_dataframe
## \notebook -draw
## Configure a Dask connection and visualize the filling of a 1D and 2D
## histograms distributedly.
##
## This tutorial showcases the process of setting up real-time data representation 
## for distributed computations.
## By calling the LiveVisualize function, you can observe the canvas updating
## with the intermediate results of the histograms as the
## distributed computation progresses. 
##
## \macro_code
## \macro_image
##
## \date August 2023
## \author Silia Taider
from dask.distributed import LocalCluster, Client
import ROOT

# Import the live visualization function
LiveVisualize = ROOT.RDF.Experimental.Distributed.LiveVisualize

# Point RDataFrame calls to Dask RDataFrame object
RDataFrame = ROOT.RDF.Experimental.Distributed.Dask.RDataFrame

# Function to create a Dask cluster and return the client
def create_connection():
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, processes=True, memory_limit="2GiB")
    client = Client(cluster)
    return client

# Function to fit a Gaussian function to the plot
def fit_gaus(plot):
    plot.Fit("gaus")

if __name__ == "__main__":
    # Setup connection to a Dask cluster
    connection = create_connection()

    # Create an RDataFrame that will use Dask as a backend for computations
    num_entries = 100000000
    d = RDataFrame(num_entries, daskclient=connection, npartitions=30)

    # Define a gaussean distribution with a variable mean
    dd = d.Define("x", f"gRandom->Gaus(10*rdfentry_/{num_entries}, 2)")\
          .Define("y", f"gRandom->Gaus(10*rdfentry_/{num_entries}, 3)")\

    # Create a 1D and a 2D histogram using the defined columns
    h_normal_1d = dd.Histo1D(("normal_1d", "1D Histogram of a Normal Distribution",
                              100, -10, 20),
                              "x")

    h_normal_2d = dd.Histo2D(("normal_2d", "2D Histogram of a Normal Distribution",
                              100, -15, 25, 
                              100, -15, 25
                              ), "x", "y")

    # Apply LiveVisualize to the histograms. 
    # The `fit_gaus` function will be applied to the accumulating partial result 
    # of the 1D histogram. The 2D histogram will not be further modified, just drawn. 
    # Find more details about usage of LiveVisualize in the RDataFrame docs.
    LiveVisualize({h_normal_1d: fit_gaus, h_normal_2d: None})
    
    # Plot the histograms side by side on a canvas
    c = ROOT.TCanvas("distrdf003", "distrdf003", 1600, 400)
    c.Divide(2, 1)
    c.cd(1)
    h_normal_1d.Draw()
    c.cd(2)
    h_normal_2d.Draw()

    c.Update()
