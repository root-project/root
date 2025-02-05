\addtogroup tutorial_dataframe
 
@{
[RDataFrame](classROOT_1_1RDataFrame.html) offers a high level interface for the analysis of data stored in [TTree](classTTree.html)s, [CSV files](classROOT_1_1RDF_1_1RCsvDS.html) and [other data formats](classROOT_1_1RDF_1_1RDataSource.html).

In addition, multi-threading and other low-level optimisations allow users to exploit all the resources available on their machines transparently.

In a nutshell:
~~~{.cpp}
ROOT::EnableImplicitMT(); // Enable ROOT's implicit multi-threading
ROOT::RDataFrame d("myTree", "file_*.root"); // Interface to TTree and TChain
auto histoA = d.Histo1D("Branch_A");  // Book the filling of a histogram
auto histoB = d.Histo1D("Branch_B");  // Book the filling of another histogram
// Data processing is triggered by the next line, which accesses a booked result for the first time
// All booked results are evaluated during the same parallel event loop.
histoA->Draw(); // <-- event loop runs here!
histoB->Draw(); // HistoB has already been filled, no event loop is run here
~~~

Explore the examples below or go to [RDataFrame's user guide](classROOT_1_1RDataFrame.html). A list of all the RDataFrame tutorials can be found [here](\ref df_alltutorials).

## Table of contents
- [Introduction](\ref df_intro)
- [Processing your data](\ref processingdata)
- [Write and read from many sources](\ref readwrite)
- [Interface with Numpy and Pandas](\ref numpypanda)
- [Distributed execution in Python](\ref df_distrdf)
- [Know more about your analysis](\ref analysisinfo)
- [Example HEP analyses tutorials](\ref hepanalysis)
- [List of all the tutorials](\ref df_alltutorials)



\anchor df_intro
## Introduction

To get started these examples show how to create a simple RDataFrame, how to process the data in a simple analyses and how to plot distributions.

| **Tutorial** || **Description** |
|------|--------|-----------------|
| df000_simple.C | df000_simple.py | Simple RDataFrame example in C++. |
| df001_introduction.C | df001_introduction.py | Basic RDataFrame usage. |
| df002_dataModel.C | df002_dataModel.py | Show how to work with non-flat data models, e.g. vectors of tracks.|


\anchor processingdata
## Processing your data

A collection of building block examples for your analysis.

| **Tutorial** || **Description** |
|------|--------|-----------------|
| df003_profiles.C | df003_profiles.py | Use TProfiles. |
| df005_fillAnyObject.C | | Fill any object the class of which exposes a Fill method  |
| df006_ranges.C | df006_ranges.py  | Use Range to limit the amount of data processed. |
| df012_DefinesAndFiltersAsStrings.C | df012_DefinesAndFiltersAsStrings.py | Use just-in-time-compiled Filters and Defines for quick prototyping. |
| df016_vecOps.C | df016_vecOps.py | Process collections in RDataFrame with the help of RVec. |
| df018_customActions.C | | Implement a custom action to fill THns. |
| df020_helpers.C | | Show usage of RDataFrame's helper tools. |
| df021_createTGraph.C | df021_createTGraph.py | Fill a TGraph. |
| df022_useKahan.C | | Implement a custom action that evaluates a Kahan sum. |
| df023_aggregate.C | | Use the Aggregate action to specify arbitrary data aggregations. |
| df025_RNode.C | | Manipulate RDF objects in functions, loops and conditional branches.|
| df036_missingBranches.C | df036_missingBranches.py | Deal with missing values due to a missing branch when switching to a new file in a chain. |
| df037_TTreeEventMatching.C | df037_TTreeEventMatching.py | Deal with missing values due to not finding a matching event in an auxiliary dataset. |


\anchor readwrite
## Write and read from many sources

The content of a dataframe can be written to a ROOT file. In addition to ROOT files, other file formats can be read.

| **Tutorial** || **Description** |
|------|--------|-----------------|
| df007_snapshot.C | df007_snapshot.py | Write out a dataset. |
| df008_createDataSetFromScratch.C | df008_createDataSetFromScratch.py | Generate data from scratch. |
| df009_FromScratchVSTTree.C | | Compare creation of a ROOT dataset with RDataFrame and TTree. |
| df010_trivialDataSource.C | df010_trivialDataSource.py | Simplest possible data source. |
| df014_CSVDataSource.C | df014_CSVDataSource.py | Process a CSV. |
| df015_LazyDataSource.C | | Concatenate computation graphs with the "lazy data source. |
| df019_Cache.C | df019_Cache.py | Cache a processed RDataFrame in memory for further usage. |
| df027_SQliteDependencyOverVersion.C | | Analyse a remote sqlite3 file. |
| df028_SQliteIPLocation.C | | Plot the location of ROOT downloads reading a remote sqlite3 file. |
| df029_SQlitePlatformDistribution.C | | Analyse data in a sqlite3 file. |
| df030_SQliteVersionsOfROOT.C | | Analyse data in a sqlite3 file and create a plot. |


\anchor numpypanda
## Interface with Numpy and Pandas

From Python, NumPy arrays can be imported into RDataFrame and columns from RDataFrame can be converted to NumPy arrays. A Pandas DataFrame can also be converted into a RDataFrame.

| **Tutorial** | **Description** |
|--------------|-----------------|
| df026_AsNumpyArrays.py | Read data into Numpy arrays. |
| df032_RDFFromNumpy.py | Read data from Numpy arrays. |
| df035_RDFFromPandas.py | Read data from Pandas DataFrame. |

\anchor df_distrdf
##Distributed execution in Python

RDataFrame applications can be executed in parallel through distributed computing frameworks on a set of remote machines via Apache Spark or Dask.

| **Tutorial** | **Description** |
|--------------|-----------------|
| distrdf001_spark_connection.py | Configure a Spark connection and fill two histograms distributedly. |
| distrdf002_dask_connection.py | Configure a Dask connection and fill two histograms distributedly. |
| distrdf003_live_visualization.py | Configure a Dask connection and visualize the filling of a 1D and 2D histograms distributedly. |

\anchor analysisinfo
## Know more about your analysis

In RDataFrame there exist methods to inspect the data and the computation graph.

| **Tutorial** || **Description** |
|------|--------|-----------------|
| df004_cutFlowReport.C | df004_cutFlowReport.py | Display cut/Filter efficiencies. |
| df013_InspectAnalysis.C | | Use callbacks to update a plot and a progress bar during the event loop. |
| df024_Display.C |df024_Display.py | Use the Display action to inspect entry values. |
| df031_Stats.C | df031_Stats.py | Use the Stats action to extract the statistics of a column. |
| | df033_Describe.py | Get information about your analysis. |
| df034_SaveGraph.C | df034_SaveGraph.py | Look at the DAG of your analysis |

\anchor hepanalysis
## Example HEP analyses tutorials

With RDataFrame advanced analyses can be executed on large amounts of data. These examples shows how particle physics analyses can be carried out using Open Data from different experiments.

| **Tutorial** || **Description** |
|------|--------|-----------------|
| df017_vecOpsHEP.C | df017_vecOpsHEP.py | Use RVecs to plot the transverse momentum of selected particles. |
| df101_h1Analysis.C | | Express ROOT's standard H1 analysis. |
| df102_NanoAODDimuonAnalysis.C | df102_NanoAODDimuonAnalysis.py | Process  NanoAOD files. |
| df103_NanoAODHiggsAnalysis.C | df103_NanoAODHiggsAnalysis.py | An example of complex analysis: reconstructing the Higgs boson. |
| | df104_HiggsToTwoPhotons.py | The Higgs to two photons analysis from the ATLAS Open Data 2020 release. |
| | df105_WBosonAnalysis.py | The W boson mass analysis from the ATLAS Open Data release of 2020. |
| df106_HiggsToFourLeptons.C | df106_HiggsToFourLeptons.py | The Higgs to four lepton analysis from the ATLAS Open Data release of 2020. |
| | df107_SingleTopAnalysis.py | A single top analysis using the ATLAS Open Data release of 2020. |

\anchor df_alltutorials
@}