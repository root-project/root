% ROOT Version 6.26 Release Notes
% 2021-03-03
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.26/00 is scheduled for release in May, 2021.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Bertrand Bellenot, CERN/SFT,\
 Josh Bendavid, CERN/CMS,\
 Jakob Blomer, CERN/SFT,\
 Rene Brun, CERN/SFT,\
 Will Buttinger, STFC/ATLAS,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/SFT,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Enrico Guiraud, CERN/SFT,\
 Jonas Hahnfeld, CERN/SFT,\
 Ivan Kabadzhov, CERN/SFT,\
 Sergey Linev, GSI,\
 Javier Lopez-Gomez, CERN/SFT,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Axel Naumann, CERN/SFT,\
 Vincenzo Eduardo Padulano, CERN/SFT and UPV,\
 Max Orok, U Ottawa,\
 Alexander Penev, University of Plovdiv,\
 Danilo Piparo, CERN/SFT,\
 Fons Rademakers, CERN/SFT,\
 Jonas Rembser, CERN/SFT,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Oksana Shadura, UNL/CMS,\
 Matevz Tadel, UCSD/CMS,\
 Vassil Vassilev, Princeton/CMS,\
 Wouter Verkerke, NIKHEF/Atlas,\
 Stefan Wunsch, CERN/SFT

## Deprecation, Removal, Backward Incompatibilities

- The "Virtual MonteCarlo" facility VMC (`montecarlo/vmc`) has been removed from ROOT. The development of this package has moved to a [separate project](https://github.com/vmc-project/). ROOT's copy of VMC was deprecated since v6.18.
- `TTreeProcessorMT::SetMaxTasksPerFilePerWorker` has been removed. `TTreeProcessorMT::SetTasksPerWorkerHint` is a superior alternative.
- `TTree::GetEntry()` and `TTree::GetEvent()` no longer have 0 as the default value for the first parameter `entry`. We are not aware of correct uses of this function without providing an entry number. If you have one, please simply pass `0` from now on.
- `TBufferMerger` is now out of the `Experimental` namespace (`ROOT::Experimental::TBufferMerger` is deprecated, please use `ROOT::TBufferMerger` instead)


## Core Libraries

### Interpreter

- As of v6.26, cling diagnostic messages can be redirected to the ROOT error handler. Users may enable/disable this via `TCling::ReportDiagnosticsToErrorHandler()`, e.g.
```cpp
root [1] gInterpreter->ReportDiagnosticsToErrorHandler();
root [2] int f() { return; }
Error in <cling>: ROOT_prompt_2:1:11: non-void function 'f' should return a value [-Wreturn-type]
int f() { return; }
          ^
```
More details at [PR #8737](https://github.com/root-project/root/pull/8737).

## I/O Libraries

- `TDirectory::WriteObject` now always saves the object's title to the file if it is derived from `TObject` (PR [#8394](https://github.com/root-project/root/pull/8934)).

### Command line utilities

- `rootls` now follows the same logic of `TFile::ls()` to print the key cycle number and its tag when listing contents of a file with the `-l` option (PR [#7878](https://github.com/root-project/root/pull/7878)):
```
$: rootls -l https://root.cern/files/ttree_read_imt.root
TTree  Mar 13 17:17 2019 TreeIMT;2 "TTree for IMT test" [current cycle]
TTree  Mar 13 17:17 2019 TreeIMT;1 "TTree for IMT test" [backup cycle]
```
- `root` will now error on receiving unrecognized options, similarly to other command line tools (PR [#8868](https://github.com/root-project/root/pull/8868)):
```
$: root --random -z --nonexistingoption
root: unrecognized option '--random'
root: unrecognized option '-z'
root: unrecognized option '--nonexistingoption'
Try 'root --help' for more information.
```

## TTree Libraries

- `TTreeReader::GetEntryStatus` now always reports `kEntryBeyondEnd` after an event loop correctly completes. In previous versions, it could sometime return `kEntryNotFound` even for well-behaved event loops.
- Add `TEntryList::AddSubList` to specifically add a sub-list to the main list of entries. Consequently, add also a new option `"sync"` in `TChain::SetEntryList` to connect the sub-trees of the chain to the sub-lists of the entry list in lockstep (PR [#8660](https://github.com/root-project/root/pull/8660)).
- Add `TEntryList::EnterRange` to add all entries in a certain range `[start, end)` to the entry list (PR [#8740](https://github.com/root-project/root/pull/8740)).

## RDataFrame

### New features

- Add `Redefine` to the `RDataFrame` interface, which allows to overwrite the value of an existing column.
- Add `Describe` to the `RDataFrame` interface, which allows to get useful information, e.g. the columns and their types.
- Add `DescribeDataset` to the `RDataFrame` interface, which allows to get information about the dataset (subset of the output of Describe()).
- Add [DefinePerSample](https://root.cern/doc/master/classROOT_1_1RDF_1_1RInterface.html#a29d77593e95c0f84e359a802e6836a0e), a method which makes it possible to define columns based on the sample and entry range being processed. It is also a useful way to register callbacks that should only be called when the input dataset/TTree changes.
- `Book` now supports just-in-time compilation, i.e. it can be called without passing the column types as template parameters (with some performance penalty, as usual).
- As an aid to `RDataSource` implementations with which collection sizes can be retrieved more efficiently than the full collection, `#var` can now be used as a short-hand notation for column name `R_rdf_sizeof_var`.
- Helpers have been added to export data from `RDataFrame` to RooFit datasets. See the "RooFit Libraries" section below for more details.
- The output format of `Display` has been significantly improved.

### Experimental Distributed RDataFrame
The distributed RDataFrame module has been improved. Now it supports sending RDataFrame tasks to a [Dask](https://dask.org/) scheduler. Through Dask, RDataFrame can be also scaled to a cluster of machines managed through a batch system like HTCondor or Slurm. Here is an example:

```python
import ROOT
from dask.distributed import Client
RDataFrame = ROOT.RDF.Experimental.Distributed.Dask.RDataFrame

# In a Python script the Dask client needs to be initalized in a context
# Jupyter notebooks / Python session don't need this
if __name__ == "__main__":
    client = Client("SCHEDULER_ADDRESS")
    df = RDataFrame("mytree","myfile.root", daskclient=client)
    # Proceed as usual
    df.Define("x","someoperation").Histo1D("x")
```

Other notable additions and improvements include:

- Greatly reduce distributed tasks processing overhead. This involved:
    - Changing the distributed execution logic with the `TTree` data source to use `TEntryList` in order to select the range of entries that each task will read from the tree. This highly reduces the waiting time spent in retrieving the correct entries for processing, as it was previously done using the `Range` operation.
    - Refactoring the internal mechanism to store information about data sources and create ranges for the distributed tasks accordingly. This will also allow in the future to easily extend the supported data sources in distributed RDataFrame.
- Refactor triggering of the computation graph in the distributed tasks, so that it now runs with the Python GIL released. This allows interoperability with frameworks like Dask that run different Python threads along the main processing one.
- Set minimum Python version to use this tool to 3.7. This allows using more modern Python functionality in distributed RDataFrame code and is in line with the Python support provided by Spark and Dask.
- Add support for the `DefinePerSample` operation.
- Fixed a bug that disregarded user provided `npartitions` parameter in distributed Spark execution.
- Improve support for friend trees in distributed executions with TTree as data source. This fixes a long-standing issue with distributed RDataFrame [#7584](https://github.com/root-project/root/issues/7584).
- Changed naming scheme of the files written by a distributed `Snapshot` operation. Now, each task will be assigned with a sequential id that will be appended to the name of the file created during the execution. For example, calling `Snapshot("mytree","myfile.root")` on a distributed RDataFrame with 3 partitions will create three files named like so: `myfile_0.root, myfile_1.root, myfile_2.root`.
- Add support for TChain data sources with no tree name and multiple different tree subnames.
- Raise an error if an in-memory-only TTree is passed as data source to a distributed RDataFrame.

### Other improvements

- The scaling to a large amount of threads of computation graphs with many simple `Filter`s or `Define`s has been greatly improved, see also [this talk](https://indico.cern.ch/event/1036730/#1-a-performance-study-of-the-r) for more details

## Histogram Libraries

- Implement the `SetStats` method for `TGraph` to turn ON or OFF the statistics box display
  for an individual `TGraph`.

## Math Libraries

- `RVec` has been heavily re-engineered in order to add a small buffer optimization and to streamline its internals. The change should provide a small performance boost to
  applications that make heavy use of `RVec`s and should otherwise be user-transparent. Please report any issues you should encounter.
- I/O support of `RVec` objects has been optimized. As a side-effect, `RVec`s can now be read back as `std::vector`s and vice-versa.
- added `ROOT::VecOps::Drop`, an operation that removes `RVec` elements at the specified indices.
- handy aliases `ROOT::RVecI`, `ROOT::RVecD`, `ROOT::RVecF`, ..., have been introduced as short-hands for `RVec<int>`, `RVec<double>`, `RVec<float>`, ...


## RooFit Libraries
### Creating RooFit datasets from RDataFrame
RooFit now contains two RDataFrame action helpers, `RooDataSetHelper` and `RooDataHistHelper`, which allow for creating RooFit datasets by booking an action:
```c++
  RooRealVar x("x", "x", -5.,   5.);
  RooRealVar y("y", "y", -50., 50.);
  auto myDataSet = rdataframe.Book<double, double>(
    RooDataSetHelper{"dataset",          // Name   (directly forwarded to RooDataSet::RooDataSet())
                    "Title of dataset",  // Title  (                   ~ " ~                      )
                    RooArgSet(x, y) },   // Variables to create in dataset
    {"x", "y"}                           // Column names from RDataFrame
  );
```
For more details, consult the tutorial [rf408_RDataFrameToRooFit](https://root.cern/doc/v626/rf408__RDataFrameToRooFit_8C.html).

### Storing global observables in RooFit datasets

RooFit groups model variables into *observables* and *parameters*, depending on if their values are stored in the dataset.
For fits with parameter constraints, there is a third kind of variables, called *global observables*.
These represent the results of auxiliary measurements that constrain the nuisance parameters.
In the RooFit implementation, a likelihood is generally the sum of two terms:
  * the likelihood of the data given the parameters, where the normalization set is the set of observables (implemented by `RooNLLVar`)
  * the constraint term, where the normalization set is the set of *global observables* (implemented by `RooConstraintSum`)

Before this release, the global observable values were always taken from the model/pdf.
With this release, a mechanism is added to store a snapshot of global observables in any `RooDataSet` or `RooDataHist`.
For toy studies where the global observables assume a different values for each toy, the bookkeeping of the set of global observables and in particular their values is much easier with this change.

Usage example for a model with global observables `g1` and `g2`:
```C++
auto data = model.generate(x, 1000); // data has only the single observables x
data->setGlobalObservables(g1, g2); // now, data also stores a snapshot of g1 and g2

// If you fit the model to the data, the global observables and their values
// are taken from the dataset:
model.fitTo(*data);

// You can still define the set of global observables yourself, but the values
// will be takes from the dataset if available:
model.fitTo(*data, GlobalObservables(g1, g2));

// To force `fitTo` to take the global observable values from the model even
// though they are in the dataset, you can use the new `GlobalObservablesSource`
// command argument:
model.fitTo(*data, GlobalObservables(g1, g2), GlobalObservablesSource("model"));
// The only other allowed value for `GlobalObservablesSource` is "data", which
// corresponds to the new default behavior explained above.
```

In case you create a RooFit dataset directly by calling its constructor, you can also pass the global observables in a command argument instead of calling `setGlobalObservables()` later:
```C++
RooDataSet data{"dataset", "dataset", x, RooFit::GlobalObservables(g1, g2)};
```

To access the set of global observables stored in a `RooAbsData`, call `RooAbsData::getGlobalObservables()`.
It returns a `nullptr` if no global observable snapshots are stored in the dataset.

### Changes in `RooAbsPdf::fitTo` behaviour for multi-range fits

The `RooAbsPdf::fitTo` and `RooAbsPdf::createNLL` functions accept a command argument to specify the fit range.
One can also fit in multiple ranges simultaneously.
The definition of such multi-range likelihoods for non-extended fits changes in this release.
Previously, the individual likelihoods were normalized separately in each range, which meant that the relative number of events in each sub-range was not used to estimate the PDF parameters.
From now on, the likelihoods are normalized by the sum of integrals in each range. This implies that the likelihood takes into account all inter-range and intra-range information.

### Deprecation of the `RooMinuit` class

The `RooMinuit` class was the old interface between RooFit and minuit. With ROOT version 5.24, the more general `RooMinimizer` adapter was introduced, which became the default with ROOT 6.08.

Before 6.26, it was possible to still use the `RooMinuit` by passing the `Minimizer("OldMinuit", "minimizer")` command argument to `RooAbsPdf::fitTo()`. This option is now removed.

### Increase of the `RooAbsArg` class version

The class version of `RooAbsArg` was incremented from 7 to 8 in this release. In some circumstances, this can cause warnings in `TStreamerInfo` for classes inheriting from `RooAbsArg` when reading older RooFit models from a file. These warnings are harmless and can be avoided by incrementing also the class version of the inheriting class.

## 2D Graphics Libraries

- Implement the option `X+` and `Y+` for reverse axis on TGraph.
- Offsets for axis titles with absolute-sized fonts (size%10 == 3) are now relative only to the font size (i.e. no longer relative to pad dimensions).
- In `TPaletteAxis` when the palette width is bigger than the palette height, the palette
  in automatically drawn horizontally.

## 3D Graphics Libraries


## Geometry Libraries


## Database Libraries


## Networking Libraries


## GUI Libraries

## WebGUI Libraries

- provide `--web=server` mode, which only printout window URLs instead of starting real web browser.
  Dedicated for the case when ROOT should be running as server application, providing different RWebWindow instances for connection.


## Montecarlo Libraries


## PROOF Libraries


## Language Bindings


## JavaScript ROOT


## Jupyter lab

- Let use created notebooks with viewers like https://nbviewer.jupyter.org/
- Fix problem with using of local JSROOT version


## Tutorials


## Class Reference Guide

- Images for ROOT7 tutorials can be generated, in json format, using the directive using
  `\macro_image (json)` in the macro header.


## Build, Configuration and Testing Infrastructure

## PyROOT

- The `ROOT` Python module is now properly serializable so that it is automatically available in the Python environment if a function or ROOT object needs to be serialized. See issue [#6764](https://github.com/root-project/root/issues/6764) for a concrete usecase.
- Improve overload resolution of functions that accept classes with long inheritance trees. Now prefer to call the function overload of the most derived class type (PR [#9092](https://github.com/root-project/root/pull/9092)).