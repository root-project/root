\defgroup tutorial_dataframe Data Frame tutorials
\ingroup Tutorials
\brief These examples show the functionalities of [RDataFrame](classROOT_1_1RDataFrame.html): ROOT's declarative analysis interface.

ROOT's [RDataFrame](classROOT_1_1RDataFrame.html) offers a high level interface for analyses of data stored in [TTree](classTTree.html)s, [CSV files](classROOT_1_1RDF_1_1RCsvDS.html) and [other data formats](classROOT_1_1RDF_1_1RDataSource.html).

In addition, multi-threading and other low-level optimisations allow users to exploit all the resources available on their machines transparently.

In a nutshell:
~~~{.cpp}
ROOT::EnableImplicitMT(); // Tell ROOT you want to go parallel
ROOT::RDataFrame d("myTree", "file_*.root"); // Interface to TTree and TChain
auto myHisto = d.Histo1D("Branch_A"); // This happens in parallel!
myHisto->Draw();
~~~

Explore the examples below or go to [RDataFrame user guide](classROOT_1_1RDataFrame.html).
