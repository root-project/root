% ROOT Version 6.42 Release Notes
% 2026-11-15
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.42.00 is scheduled for release in November 2026.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Bertrand Bellenot, CERN/EP-SFT,\
 Jakob Blomer, CERN/EP-SFT,\
 Lukas Breitwieser, CERN/EP-SFT,\
 Philippe Canal, FNAL,\
 Olivier Couet, CERN/EP-SFT,\
 Marta Czurylo, CERN/EP-SFT,\
 Florine de Geus, CERN/EP-SFT and University of Twente,\
 Andrei Gheata, CERN/EP-SFT,\
 Jonas Hahnfeld, CERN/EP-SFT and Goethe University Frankfurt,\
 Fernando Hueso Gonzalez, IFIC (CSIC-University of Valencia),\
 Stephan Hageboeck, CERN/EP-SFT,\
 Aaron Jomy, CERN/EP-SFT,\
 Sergey Linev, GSI Darmstadt,\
 Lorenzo Moneta, CERN/EP-SFT,\
 Vincenzo Eduardo Padulano, CERN/EP-SFT,\
 Giacomo Parolini, CERN/EP-SFT,\
 Danilo Piparo, CERN/EP-SFT,\
 Jonas Rembser, CERN/EP-SFT,\
 Silia Taider, CERN/EP-SFT,\
 Devajith Valaparambil Sreeramaswamy, CERN/EP-SFT,\
 Vassil Vassilev, Princeton,\
 Sandro Wenzel, CERN/EP-ALICE,\
 Tristan Wenzel, ETHZ,\

## Deprecation and Removal

* The `rmkdepend` executable is deprecated and will be removed from the public executables shipped with the installation of ROOT 6.44. Use instead `CMake` for dependency tracking, or `makedepend` from system packages. If you need manual inspection, on `g++` or `clang`, a similar result can be achieved by using e.g. `-std=c++17 -MM -MF file.d -MT -I incs defines -x c++-header`; on Windows with  `/std:c++17 /nologo /E /showIncludes /I incs defines /TP`.
* The outdated Windows scripts `misc/win/ld.sh`, `misc/win/makelib.sh` and `roottest/scripts/winmakelib.sh` are removed in ROOT 6.42.
* The `rootql` and `rootsl` (QuickLook and Spotlight) plugins for visualizing ROOT files in MacOS, that were developed for ROOT 5.24 are no longer supported in ROOT 6 and are thus removed from the codebase.
* The build options `vc`, `veccore`, `builtin_vc`, `builtin_veccore` and `rpath` that were deprecated are now removed and will result in configuration errors if used.
* The option `fail-on-missing=OFF` is no longer honored for opt-in (ie OFF by default) build options requiring external dependencies such as `arrow`, `cocoa`, `daos`, `daos_mock`, `dcache`, `experimental_adaptivecpp`, `fcgi`, `fortran`, `gviz`, `mpi`, `pythia8`, `qt6web`, `tmva-cudnn`, `tmva-pymva`, `tmva-sofie`, `uring` or `vecgeom`. If the respective associated package dependency is not installed, ROOT will always raise a configuration error independent of the value of `fail-on-missing`. The user has to take action by either providing the dependency or manually disabling that option via `-Darrow=OFF`.
  Note that `all=ON` enables several of these options, so building with `-Dall=ON` now requires all of their dependencies to be installed, or the unwanted ones to be disabled explicitly.
  Build options that are enabled by default, such as `pyroot`, `opengl`, `xml`, `sqlite`, `davix`, `curl`, `tmva-cpu` or `tpython` are not affected: they are still disabled automatically when their dependencies are missing.
* The option `fail-on-missing=OFF` will no longer be honored for CMake ROOT build options that have easy-to-install dependencies (e.g. via homebrew or apt-get), such as those required by options `cfitsio`, `civetweb`, `fftw3`, `imt`, `mathmore`, `nlohmann_json`, `tmva-cpu`, `unuran`, `vdt` or `xrootd`. Before, associated `builtin_option` was automatically turned ON (or the opt-in feature turned to OFF), now, user has to install system package or manually set `builtin_option` to `ON` or opt-in feature to `OFF`.
* The use of `CINT`-like implicit declarations such as `i = 0` is deprecated since [ROOT 6.32](https://root.cern/doc/v632/release-notes.html#deprecation-and-removal) and emits warnings `declaration without the 'auto' keyword is deprecated`. ROOT 6.44 will be the last release to support this syntax, after ROOT 6.44, the warning will turn into an error. Update those declarations by explicitly using the `auto` keyword: `auto i = 0`.
* The legacy evaluation backend of RooFit and the related `RooFit::BatchMode()` command argument are deprecated and will be removed in ROOT 6.44. See the RooFit section below for details.
* The method `RooRealVar::removeRange()` and the corresponding method in `RooErrorVar` that were deprecated in ROOT 6.40 are now removed.
* The overloads of `RooAbsReal::createChi2()` and `RooAbsReal::chi2FitTo()` that take unbinned **RooDataSet** data objects were deprecated in ROOT 6.40 and are now removed.
* The **RooStats::HybridPlot** class and the related **HybridResult::GetPlot** method were deprecated in ROOT 6.40 and are now removed.
* The `builtin_zeromq` and `builtin_cppzmq` build options that were deprecated in ROOT 6.40 are now removed.
* The `roofit_multiprocess` build option is deprecated and will be removed in ROOT 6.44. It has no effect anymore: RooFit's multi-process test statistics no longer depend on ZeroMQ and are now always built on non-Windows platforms, so there is no reason for an opt-in build option anymore (see the RooFit section below).
* The ROOT **auth** package together with `TVirtualAuth` and `TROOT::GetListOfSecContexts()`, and the **authenticated sockets** (`TSocket::CreateAuthSocket()`) feature are now removed following deprecation in ROOT 6.40.
* The `TSSLSocket` class is now removed following deprecation in ROOT 6.40.
* The bindings to the R programming language that are enabled with the `r=ON` or `tmva-rmva=ON` build options (`TRInterface`, RMVA, and friends) are removed, following deprecation in ROOT 6.40. Their maintenance is no longer justified, given the broader adoption of the scientific Python ecosystem. Users who still rely on R from C++ are encouraged to call R directly via https://cran.r-project.org/package=RInside, which is what the ROOT bindings were using internally.
* Several enums that are redundant with `ROOT::ESTLType` are deprecated and will be removed in ROOT 6.44: `TClassEdit::ESTLType`, `TDictionary::ESTLType`, `TStreamerElement::ESTLType`. Please use `ROOT::ESTLType` instead.
* The Makefile templates for external projects (`etc/Makefile.arch` and the files generated from `config/Makefile-comp.in` and `config/Makefile.in`) are removed. External projects should migrate to CMake (for example with `find_package(ROOT)`), or obtain compilation and linker flags from `root-config --cflags --libs`.
* The conversion from Python set to **RooArgSet** is deprecated and won't work anymore in ROOT 6.44. The problem is that Python sets are unordered while RooArgSets are ordered, and this mismatch can lead to subtle problems later on. Prefer conversion from Python lists or tuples, which are ordered too.
* The **TMPIFile** class and the `mpi` build option (not to be confused with `minuit2_mpi`, which is unaffected) are deprecated and will be removed in ROOT 6.44.
* The ROOT IO capability for the `TMVA::Experimental::SOFIE::RModel` has been removed. Users should not be encouraged to serialize models in experimental classes. For the serialization of ONNX models one can already use ONNX directly, and even serialize the ONNX bytes to a ROOT file if required.
* The ROOT binary weight file format of SOFIE (`TMVA::Experimental::SOFIE::Options::kRootBinaryWeightFile`, `WeightFileType::RootBinary`) is removed. In its place, weights can now be written in the binary [safetensors](https://huggingface.co/docs/safetensors) format with `TMVA::Experimental::SOFIE::Options::kSafetensorsWeightFile`: the payload is stored as raw little-endian data behind a small JSON header, loads faster, round-trips the values bit-exactly, and can be inspected with the standard Python and Rust safetensors tooling. Sessions of models generated with this option can also be constructed directly from an in-memory safetensors blob. The default text weight format (`*.dat`) is unchanged.
* The Keras and PyTorch parsers for SOFIE (`TMVA::Experimental::SOFIE::PyKeras` and `PyTorch`) are now removed, so `RSofieReader` only accepts ONNX files.
These parsers relied on private implementation details of Keras and PyTorch, which change faster than is appropriate for ROOT's stability standards.
Users are encouraged to export their models to ONNX and use the retained ONNX parser instead.
* **PyMVA**, the TMVA interface to Python machine-learning libraries (the `PyKeras`, `PyTorch`, `PyRandomForest`, `PyGTB` and `PyAdaBoost` methods), and the corresponding `tmva-pymva` build option are deprecated and will be removed in ROOT 6.44. Like the SOFIE Keras and PyTorch parsers, PyMVA relies on implementation details of the underlying Python libraries that change faster than is appropriate for ROOT's stability standards. Users are encouraged to train and evaluate their models directly with the Python machine-learning libraries, which integrate well with ROOT via the `ROOT::Experimental::ML::DataLoader`. For high-performance inference in C++, models can be exported to ONNX and evaluated with SOFIE (see `RSofieReader`).
* The graph_nets-based SOFIE GNN support (`TMVA::Experimental::SOFIE::RModel_GNN`, `RModel_GraphIndependent`, the `RFunction` classes and the corresponding `ParseFromMemory` Python functions) is removed.
It could only parse models built with DeepMind's *graph_nets* and *dm-sonnet* Python packages, which have been unmaintained since 2020 and can no longer be installed alongside current Python and TensorFlow versions; the parser also relied on private implementation details of those packages.
The same graph-network models (following the formalism of Battaglia et al., [arXiv:1806.01261](https://arxiv.org/abs/1806.01261)) can be defined in PyTorch and deployed through the retained SOFIE ONNX parser, whose operator support (`Gather`, `ScatterElements` with add reduction, `ReduceSum`, `LayerNormalization`, ...) covers the graph-network building blocks, including graphs with a variable number of nodes and edges.
The `TMVA_SOFIE_GNN` tutorials have been migrated to this workflow and produce inference results identical to the removed implementation.
* The ROOT IO capability for the `TMVA::Experimental::RBDT` class has been removed, along with the `TMVA.Experimental.SaveXGBoost` Python function. Experimental classes should not be persistified since their on-disk layout is not guaranteed to be stable. An `RBDT` is now built directly from an XGBoost model in its native JSON serialization with the new `TMVA::Experimental::RBDT::LoadXGBoost(jsonPath)`, which works both from C++ and Python. To convert a trained model, save it first with XGBoost's `Booster.save_model("model.json")` and then load it with `LoadXGBoost`.
* The **JsMVA** feature for interactive TMVA training in Jupyter notebooks is now removed. It was not functional for years and was therefore already excluded from ROOT 6.38. This also removes the `TMVA::IPythonInteractive` class and the related interactive-training interfaces from the TMVA method and fitter classes, such as `MethodBase::ExitFromTraining()` or `FitterBase::SetIPythonInteractive()`.
* The `TMVA::Experimental::RStandardScaler` has been removed. It computed means and standard deviations, which can be easily done without this class.
* The `TMVA::Experimental::Classification` and `TMVA::Experimental::ClassificationResult` classes are removed. They were the two-class front-end of the 2016–2017 `TMVA::Envelope` modernization campaign that never graduated out of the experimental namespace and saw no adoption. The parts of that effort that did find adoption remain available: `TMVA::Envelope`, `TMVA::CrossValidation`, `TMVA::VariableImportance`, `TMVA::HyperParameterOptimisation` and `TMVA::ROCCurve`, the latter providing the ROC curves that were previously obtained via `ClassificationResult`.
* The `TMVA::Experimental::RTensor` class was removed, since it was not much adopted by ROOTs own interfaces and is also getting redundant with `std::mdspan` in C++23.
* The **RooStats::DebuggingSampler** and **RooStats::DebuggingTestStat** classes are removed. They were mock implementations of the `TestStatSampler` and `TestStatistic` interfaces that returned uniform random numbers independent of the data, only meant for debugging the RooStats framework itself during its initial development.
* Flattening nested **RooSimultaneous** in the constructor is deprecated and will be removed in ROOT 6.44. It double counts extended components that don't depend on all index categories ([GitHub issue #23342](https://github.com/root-project/root/issues/23342)) and breaks nested RooSimultaneous whose index category is a parameter ("switch mode"). From ROOT 6.44, nested RooSimultaneous are only supported in switch mode. To combine channels, use one RooSimultaneous over a **RooCategory** with one state per channel. Workspaces read from files are not affected.
* The `RooTrace` class is deprecated and will be removed in ROOT 6.44. It was a RooFit-specific memory tracer whose instrumentation hooks are compiled out by default, so it has been inert and untested for years. For memory debugging, please use general-purpose tools such as AddressSanitizer or Valgrind instead.
* Support for the AIX operating system has been removed from the codebase. This support has not been tested since the late v5 releases and the LLVM JIT is not yet supporting AIX.
* The target `RCsg` (a library mixture of CSGLib and [SOLID](https://github.com/dtecta/solid3)) is now deprecated and will be fully private after ROOT 6.44, meaning the header `CsgOps.h` with the class `RootCsg::TBaseMesh` will no longer be public. Use instead `TGeoTessellated` public interface. If that public interface doesn't provide what you need, contact the ROOT team to request its extension.
* The `ROOT::Math::ParamFunctionBase`, `ROOT::Math::ParamFunctorHandler` and `ROOT::Math::ParamMemFunHandler` classes in `Math/ParamFunctor.h` are removed, together with the `ParamFunctor::GetImpl()` and `ParamFunctor::SetFunction()` methods that exposed them. They implemented the type erasure that `ParamFunctor` now gets from `std::function`, mirroring what was already done for `ROOT::Math::Functor`. Constructing and calling a `ParamFunctor` is unchanged, except that the constructor from an object and one of its member functions now takes a plain pointer to the object instead of anything dereferenceable, so smart pointers are no longer accepted there.
* The header `compiledata.h` will be renamed in ROOT 6.44. It did not have header guards and is not meant to be used directly. It is solely used by ROOT internally to setup `ACLiC`, consider using `gSystem` instead to access its content.
* Including `RConfig.h` and `RVersion.h` is now deprecated and will be removed after ROOT 6.44, use instead `ROOT/RConfig.hxx` and `ROOT/RVersion.hxx`.
* The headers of the `bvh` library in `Geom` are no longer public in ROOT 6.42.
* Usage of the following headers is now deprecated. These will be removed during the lifetime of ROOT 7:
  * `Rpair.h`: use `<utility>` instead.
  * `Hparam.h` and `Hoption.h`: use instead `THistPainter`, `TGraph2DPainter` public interfaces.
  * `Htypes.h` and `Gtypes.h`: Use instead `Rtypes.h`.
  * `Bswapcpy.h`: no replacement.
  * `VectorizedTMath.h`: use instead `TMathVectorized.h`.
  * `PosixThreadInc.h`: use instead `<ctime>` and `<cstdlib>`.
  * `RStringView.h`: use `<string_view>` or `ROOT/RStringView.hxx` instead.
  * `NetErrors.h`: Use instead `TSocket` public interface.
  * `strlcpy.h` and `strtok.h`: are deprecated (but will not emit warnings) and will no longer be part of ROOT installed headers in ROOT 6.44.
  * `snprintf.h`: use instead `<cstdio>`.
  * `Strlen.h`: use `<cstring>` directly as a replacement. `NEED_STRING` macro should not be defined or an error will be raised.
  * `HelpTextTV.h`: no replacement.
  * `Varargs.h` and the macro `R__VA_COPY`: use `<cstdarg>` instead.
  * `TreeUtils.h`: use instead `TNtuple::ReadStream` or `TNtupleD::ReadStream`.
  * `Riostream.h`: use `<iostream>`, `<fstream>` or `<iomanip>` instead.
  * `Rstrstream.h`: use instead `<sstream>`.
  * `ZipLZMA.h`, `ZipLZ4.h` and `ZipZSTD.h`: use instead the public methods in the `RZip.h` interface.
  * `Hepevt.h` and `cfortran.h`: these will be moved to `ROOT/Hepevt.h` and `ROOT/cfortran.h` in ROOT 6.44. At a later time point they may be fully removed from ROOT public headers, consider using instead the `TGenerator.h` public interface or the `cfortran.h` header of the `cfitsio` library, respectively.
* Remove outdated `TSQLFile` class, was not tested for very long time, not working with only supported sqlite DB

## Build System

### CMake version

CMake versions below 3.26 will emit a warning during configure, since they will become obsolete soon: CMake 3.26 will be the minimum required in ROOT 6.44. It is the version shipped with RHEL 8.10, which is the oldest Linux distribution supported by that release.

After ROOT 6.44, the minimum may rise to 3.31 if an upgrade to LLVM 24 or later is performed, see the [minimum version of CMake required by LLVM](https://discourse.llvm.org/t/rfc-raising-minimum-required-cmake-version-to-3-31/91086).


### Optimization of ROOT header files

In ROOT 6.22, many (but not all) unused includes were removed from ROOT header files. The remaining unused includes will be removed after ROOT 6.44.
For instance, `#include "TBuffer.h"` will be removed from `TKey.h`. This change may cause errors during compilation of ROOT-based code if one was implicitly relying on this transitive include on downstream code using `TBuffer` without actually including `TBuffer.h`. Another example: `TStyle.h` no longer includes internally `TArrayI.h`.
To fix it well in advance on your side in downstream code, since no warnings are emitted in the meantime, we recommend getting a `preview` of those to-be errors by adding in your code `#define R__LESS_INCLUDES`, adding this at the very top before including any ROOT header. An alternative is to define this on CMake side via `target_compile_definitions`.
This may also improve compile time and reduce code inter-dependency; see https://github.com/include-what-you-use/include-what-you-use/blob/master/docs/WhyIWYU.md for a good overview of the motivation.
The macro `R__LESS_INCLUDES` will no longer have an effect after ROOT 6.44 since it will be the new default behavior.
Note: if you build ROOT with option `dev=ON`, the header `RConfigure.h` will already contain that definition, so in that case consider just including `RConfigure.h` at the very top, rather than redefining `R__LESS_INCLUDES` in downstream code, or alternatively guard the definition with `#ifndef`.

### Moving from builtin dependencies to system-provided packages

* The general direction of the ROOT project is to become more and more reliant on system packages. It is *recommended* to make the packages required by ROOT available on the system, e.g. via a package manager, and not with the builtin mechanism. This allows for timely updates and reduces the size of the installed binaries.
* The previously vendored builtins `ftgl`, `gl2ps`, `gtest`, `nlohmann_json`, `unuran`, `civetweb`, `xxhash`, `pcre2`, should be installed in the system if possible (e.g. via `apt-get` or `homebrew` package managers). ROOT will not automatically fall-back to their builtin versions if these are not found: the user is informed of that with a helpful message. If installing these dependencies in the system is not possible, the CMake option `-Dbuiltin_XYZ=ON` has to be consciously chosen by the user.
* For the builtin versions of `ftgl`, `gl2ps`, `gtest`, `nlohmann_json`, `unuran`, `civetweb`, `xxhash`, `pcre2`, the source tarballs are now fetched from [SPI](https://spi.web.cern.ch)'s [website](https://lcgpackages.web.cern.ch/), as for the vast majority of ROOT's builtins.

## Python Interface

ROOT dropped support for Python 3.10, meaning ROOT now requires at least Python 3.11.

### Connecting Python callables to signals

`TQObject::Connect()` now directly accepts a Python callable as the slot, for
example `button.Connect("Clicked()", on_clicked)`. The arguments emitted by the
signal are forwarded to the callable, as far as its signature accepts them, and
the connection keeps the callable alive. Use `Disconnect(signal, callable)` to
undo the connection. Signals of any signature are supported, no longer only
those covered by the `TPyDispatcher::Dispatch()` overloads.

The `TPyDispatcher` class and its `ROOT/TPyDispatcher.h` header are removed:
it required the user to create and keep alive a dispatcher object manually,
and it was broken in recent releases anyway, since the interpreter could not
resolve its symbols from the `libROOTPythonizations` Python extension module.
Replace `obj.Connect(signal, "TPyDispatcher", disp, "Dispatch()")` with
`obj.Connect(signal, callable)`.

### Pure Python callables in RDataFrame

A Python function passed to `RDataFrame.Define()` or `Filter()` now runs as it is in the Python interpreter instead of being compiled with Numba behind the scenes.
It can therefore use any Python code, including ROOT classes and their methods, and **any Python library**.
The function needs type annotations for its parameters and its return value, which RDataFrame reads to know their C++ types: these can be C++ type names, Python types (`bool`, `int`, `float`, `str`) or ROOT classes.
As before, if no column list is given, the parameters are read from the columns with the same names.

Usage:
```python
from particle import Particle

def mass_difference(p: ROOT.Math.PtEtaPhiMVector, pdg: int) -> float:
    return p.M() - Particle.from_pdgid(pdg).mass / 1000

df = df.Define("dm", mass_difference)  # reads the columns `p` and `pdg`
```

Numba is now opt-in: a function decorated with `ROOT.Numba.Declare` is still JIT compiled with Numba, as before.

```python
@ROOT.Numba.Declare(["double"], "double")
def square(x):
    return x * x

df = df.Define("x2", square, ["x"])
```

Behavior changes:
* A Python callable without type annotations, e.g. a lambda, now raises a `TypeError`: annotate its parameters and return value, or decorate it with `ROOT.Numba.Declare`.

## Machine Learning integration

`ROOT.Experimental.ML.RDataLoader` can now read files directly, save its training and validation datasets to disk and return batches as JAX arrays. Loading is faster and uses less memory.

### File paths as input
`RDataLoader` can now be created directly from files without building the RDataFrame first if the dataset doesn't need to be manipulated before training.
A `(dataset_name, files)` tuple reads the TTree or RNTuple `dataset_name` from one file path or a list of them.

Usage:
```python
dl = ROOT.Experimental.ML.RDataLoader(("events", ["data_1.root", "data_2.root"]), batch_size=1024, target="label")
```

### Saving the training and validation datasets
The new `save()` method writes a dataset to disk as it would be loaded for training, i.e. after filtering, splitting, shuffling and padding, as a TTree (default) or an RNTuple.

Usage:
```python
train, val = dl.train_test_split(test_size=0.2)

train.save("tree", "train.root")
val.save("tree", "val.root", output_format="rntuple")
```

### JAX output format
Batches can now be returned as [`jax.Array`](https://docs.jax.dev/en/latest/_autosummary/jax.Array.html) objects with `as_jax()`, which takes an optional `device` parameter like `as_torch()`.

### Performance improvements
The loader no longer needs to be just-in-time compiled for the column types of each dataset, as the columns are now read by a custom RDataFrame action.
The eager loading path avoids most copies of the dataset, and both loading modes read the data in fewer event loops.

### Interface changes
* All `RDataLoader` arguments but the inputs are now keyword-only, e.g. `RDataLoader(df, batch_size=64)` instead of `RDataLoader(df, 64)`, and the first argument is renamed from `rdataframes` to `inputs`.
* `RDataLoader` does not support `ImplicitMT` yet: a `RuntimeError` is now raised if it is enabled when the input RDataFrames or the loader are created.


## I/O

* Reading a collection without its dictionary no longer crashes when the elements hold a `std::string` or a `TString`. The emulated collection proxy relocated its elements with a raw memory copy when its buffer had to grow, which corrupts an object that points into itself, such as a `std::string` using the small string optimization; the invalid pointer was then freed when the object was destroyed. Such elements are now destroyed and reconstructed at the new location instead. This affected for instance a `std::vector<std::pair<std::string,double>>` read back without a dictionary.

* An `RNTupleReader` will now by default not load all metadata on open. Metadata related to the on-disk location of data pages will instead be loaded during iteration. To restore the old behavior, set the metadata mode in the `RNTupleReadOptions` to `kEager`.

### RFile

RFile, ROOT's modern interface to ROOT file, has moved out of the `ROOT::Experimental` namespace.

### RNTupleProcessor

The RNTupleProcessor is a new, standalone interface for creating *chains* (similar to TTree's `TChain`) and *joins* (similar to TTree friends) from RNTuples in a composable way, and iterating over their entries.
Until this release, it was available in the experimental namespace.
With ROOT v6.42:

* The RNTupleProcessor for single RNTuples and chained RNTuples has moved out of `ROOT::Experimental`.
They can be created with `ROOT::RNTupleProcessor::Create()` and `ROOT::RNTupleProcessor::CreateChain()`, respectively.
Processing joined RNTuples will remain available in the experimental namespace for this release, in order to address a few additional (potentially interface-breaking) improvements.
Creating an RNTupleProcessor for joins can (for now) be done by calling `ROOT::Experimental::CreateRNTupleJoinProcessor`.
* An experimental RDataFrame datasource using the RNTupleProcessor has been added.
RDataFrames can be created from an existing RNTupleProcessor by calling `ROOT::Experimental::RDF::FromRNTupleProcessor()`.

### RNTuple SoA I/O

The RNTuple SoA I/O classes moved out of experimental. These classes and the related dictionary options provide a mechanism to read and write a collection of a class, i.e. an "array of struct" (AoS),
as "struct of array" (SoA) in memory. This is useful, e.g., for efficient consumption of data structures by GPUs. This new feature comes with a technical documentation (`SoA.md`) and a tutorial (`ntpl020_SoA.C`).

## Core

* `TClass::IsTriviallyRelocatable()` reports whether an object of a class can be moved to a new address with a raw memory copy, i.e. without running a move or copy constructor (trivial relocatability in the C++26 sense). It is backed by the new `kClassIsTriviallyRelocatable` class property, which `TInterpreter::ClassInfo_ClassProperty()` now fills in. A class the interpreter does not know about, in particular an emulated one, is conservatively reported as not relocatable.

## Histograms

### Cumulative histograms in more than one dimension

`TH1::GetCumulative()` now computes a true multi-dimensional cumulative for 2D
(`TH2`) and 3D (`TH3`) histograms, using the inclusion-exclusion principle: each
bin of the result holds the sum of all bins whose indices are no greater than
(forward) or no less than (backward) those of the target bin along *every* axis.
Previously the method accumulated a single running sum over the flattened bin
iteration, which did not correspond to a meaningful cumulative distribution in
more than one dimension.

The behavior for one-dimensional histograms is unchanged. Code that relied on
the previous 2D/3D output (for example to build per-axis selection efficiency
maps) will now obtain different, mathematically consistent values.

### New histogram package

The new histogram package continues development in the `ROOT::Experimental` namespace.
This release introduces `RHistEngine::SnapshotAtomic` to obtain a consistent copy during concurrent filling.
There is now a class [`ROOT::Experimental::RProfile`](https://root.cern/doc/v642/classROOT_1_1Experimental_1_1RProfile.html)
for (multidimensional) profile histograms.
`RHist` and `RHistEngine` with two or three dimensions can be converted to `TH2` or `TH3`.
More conversion functions will continue to be added in the future (`THn`, `TProfile`).

## Math

### Skipping identically-vanishing second derivatives in numerical Hessian evaluation

Minuit2 objective functions can now advertise pairs of parameters whose mixed second derivative is identically zero, via the new virtual function `ROOT::Minuit2::FCNBase::SecondDerivativeAlwaysVanishes()` (with corresponding setters on `ROOT::Minuit2::FCNAdapter` and `ROOT::Minuit2::Minuit2Minimizer`).
The numerical Hessian computation in Minuit2 (`MnHesse`) skips the finite-difference evaluations for such parameter pairs, which can speed up Hesse significantly for likelihoods with many mutually independent parameters.
The parameter indices in this interface always refer to the function's own full (external) parameter space, including parameters that are fixed in the minimizer, and the advertised information must hold for all parameter values; see the `FCNBase` documentation for the full contract.

## RDataFrame

* Added `RedefinePerSample` transformation. Works similarly to `DefinePerSample`, but allows to redefine existing values
  of a column on a per-sample basis. This operation is supported in local and distributed mode.
* Added `Median` action. Computes the exact median of the input column. At the moment this action is supported in local
  mode only.
* Python callables passed to `Define` and `Filter` now run in the Python interpreter, and Numba compilation is opt-in,
  see [Pure Python callables in RDataFrame](#pure-python-callables-in-rdataframe).

## Trees

### Behavior change: `sqrt()` of negative arguments in TTreeFormula now returns NaN

Since its introduction in 1995, the formula engine used by `TTree::Draw()`, `TTree::Scan()` and `TTreeFormula`
silently evaluated `sqrt(x)` as `sqrt(abs(x))` for negative arguments (or as `0` in the optimized evaluation path
of the legacy `ROOT::v5::TFormula`). This could produce silently wrong results, e.g. in selections involving
`sqrt` of an expression that can become negative. `sqrt()` now returns NaN for negative arguments, consistent
with `TMath::Sqrt()`, the standard C `sqrt()`, and the modern `TFormula` used by `TF1`.
Note that in a selection, a NaN evaluates as `false`, so entries where the `sqrt` argument is negative now fail
the cut instead of being selected based on `sqrt(abs(x))`.

## RooFit

### RooFit::MultiProcess without ZeroMQ, now enabled by default

The `RooFit::MultiProcess` package that implements the parallel gradient minimization with `fitTo(..., RooFit::Parallelize(n))` previously communicated between the forked processes with ZeroMQ sockets, which required building ROOT with `roofit_multiprocess=ON` and the ZeroMQ (with draft API) and cppzmq dependencies.
The interprocess communication is now implemented directly on top of plain `socketpair()` pipes that are inherited by the forked worker processes, so the ZeroMQ and cppzmq dependencies and the `RooFitZMQ` library are removed entirely.
Since the feature no longer needs extra dependencies, it is now always built on non-Windows platforms and the `roofit_multiprocess` build option has no effect anymore; it is deprecated and will be removed in ROOT 6.44.
For implementers of custom `RooFit::MultiProcess::Job` subclasses, the message type in the `Job` interface changed from `zmq::message_t` to the new `RooFit::MultiProcess::Message` byte-buffer class, which supports the same usage patterns.

### Small changes

* The `RooMinimizer::Strategy` enum has been removed. It named the Minuit strategies that are usually referred to just by integers, but caused confusion because it didn't include the unnamed "Strategy 3". Since people usually set the strategy with integer values anyway, it was decided that the simplest solution to avoid the confusion was simply to remove the `RooMinimizer::Strategy` enum

* `RooSimultaneous::expectedEvents()` with a **RooSuperCategory** index now sums over all components if its input categories are in the normalization set, as for dataset variables. Before, it returned the selected component's yield, which also gave the wrong total in extended toy generation.

* An extended **RooSimultaneous** whose index category is a parameter ("switch mode", like **RooMultiPdf**) now evaluates to the selected component and uses its expected events in the extended term, also with code generation. Before, the value was weighted by the component's relative yield, and the new backends dropped the extended term. Consequently, `RooSimultaneous::getVal(normSet)` only includes the relative channel yields if the index category is in `normSet`.

### Faster Hesse for likelihoods with many independent parameters

RooFit now analyzes the computation graph of the minimized function to find pairs of parameters that never appear in the same additive term of the likelihood, meaning their mixed second derivative is identically zero.
This information is forwarded to Minuit2, which skips the corresponding finite-difference evaluations in the numerical Hessian computation (see the Math section above).
For likelihoods with many mutually independent parameters, such as the per-channel nuisance parameters of large combined HistFactory models, this can speed up `RooMinimizer::hesse()` by 30 % or more, with results identical up to floating-point noise.
This optimization is automatic and requires no user action.

### Deprecation of the legacy evaluation backend

The `legacy` evaluation backend for likelihood and chi-square fits is deprecated and will be removed in ROOT 6.44.
It was superseded by the vectorized `cpu` backend, which is the default since ROOT 6.32.
After the removal of the constant term optimization (see below), the legacy backend also has no performance-relevant feature left that would justify its continued maintenance.

Selecting the legacy backend with `RooFit::EvalBackend("legacy")` now prints a deprecation warning whenever a likelihood or chi-square object is created with it, and the `RooFit::EvalBackend::Legacy()` factory function is marked as deprecated, resulting in compiler warnings.

The **RooFit::BatchMode()** command argument, which was superseded by `RooFit::EvalBackend()` in ROOT 6.28, is deprecated at the same time and will also be removed in ROOT 6.44.
Note that the C++ declarations of `RooFit::BatchMode()` had been unintentionally absent since ROOT 6.30; they are restored in this release, marked as deprecated, to give downstream code a proper migration window.

The removal in ROOT 6.44 will also include:

  * the implementation classes of the legacy test statistics: **RooNLLVar**, **RooChi2Var**, **RooAbsOptTestStatistic** and **RooAbsTestStatistic** (their headers are not part of the public interface anymore since ROOT 6.32, but they are still installed),
  * the old multiprocessing mechanism of the legacy backend, consisting of the **RooRealMPFE** class and the underlying **BidirMMapPipe**,
  * the `nll::name[pdf,data]` and `chi2::name[pdf,data]` expressions in the `RooWorkspace::factory()` language, which instantiate the removed classes directly.

Users are strongly encouraged to switch to the default `cpu` evaluation backend, i.e., to simply not pass any `EvalBackend()` or `BatchMode()` command argument.
If the default backend does not work for a given use case, **please report it by opening an issue on the ROOT GitHub repository**.

### Removal of the constant term optimization for legacy test statistic classes

The **RooFit::Optimize()** option (constant term optimization) was deprecated in ROOT 6.40, and its functionality is now removed.
The `RooFit::Optimize()` and `RooMinimizer::optimizeConst()` methods are kept for API consistency across ROOT versions, but they have no effect anymore.

In practice this option only affected the `legacy` evaluation backend.

Together with the mechanism itself, the public interfaces that only existed to
drive it were removed. Code that called or overrode any of the following needs
to be adapted:

  * `RooAbsArg::constOptimizeTestStatistic()`, `RooAbsArg::findConstantNodes()`,
    `RooAbsArg::setCacheAndTrackHints()`, `RooAbsArg::canNodeBeCached()` and the
    `RooAbsArg::ConstOpCode` and `RooAbsArg::CacheMode` enums.
  * `RooAbsData::cacheArgs()`, `resetCache()`, `setArgStatus()`, `attachCache()`,
    `optimizeReadingWithCaching()`, `allClientsCached()` and `hasFilledCache()`,
    together with the corresponding `RooAbsDataStore` interface (`cacheArgs()`,
    `cacheOwner()`, `attachCache()`, `setArgStatus()`, `resetCache()`,
    `cachedVars()`, `recalculateCache()` and `forceCacheUpdate()`). Classes
    deriving from `RooAbsDataStore` no longer need to implement them.
  * `RooFit::TestStatistics::RooAbsL::constOptimizeTestStatistic()` and
    `RooFit::TestStatistics::LikelihoodWrapper::constOptimizeTestStatistic()`.

Since the cache-and-track hints are gone, the `"CacheAndTrack"`,
`"NOCacheAndTrack"` and `"NeverConstant"` attributes no longer have any effect.

The default vectorized CPU evaluation backend (introduced in ROOT 6.32) already performs these optimizations automatically and is not affected by this change.
Users are strongly encouraged to switch to the vectorized CPU backend if they are still using the legacy backend.

If the vectorized backend does not work for a given use case, **please report it by opening an issue on the ROOT GitHub repository**.

### Default binning of RooFit variables changed to zero bins

A freshly-constructed `RooRealVar` (or `RooErrorVar`) no longer has a default binning of 100 bins.
Instead, `RooAbsRealLValue::getBins()` now returns `0` until a binning is explicitly set (e.g. via `RooRealVar::setBins()`).
This makes it possible to distinguish a variable whose binning was deliberately chosen from one that was left at the default, which avoids
writing redundant `nbins` fields when serializing workspaces to HS3 JSON.

For the cases that previously relied on the default of 100 bins, that value is now injected by the relevant routine:
unbinned-dataset plotting (`RooAbsRealLValue::frame()`), `RooAbsPdf::generateBinned()`, and `RooAbsRealLValue::createHistogram()`
all fall back to `RooAbsRealLValue::DefaultNBins` (100) when the variable has no binning set.
As a result, plotting, generating binned data and creating histograms from a default-constructed variable behave exactly as before.

Code that reads `getBins()`/`numBins()` of a bare variable and expected the value `100` should either set the binning explicitly,
or read the bin count from the relevant histogram or plot frame instead.

### Resolution models are no longer imported into the workspace as standalone objects

When a `RooResolutionModel` (like `RooGaussModel` or `RooTruthModel`) is used wi th a
`RooAbsAnaConvPdf` (like `RooDecay`, `RooBDecay`, etc.), it acts as a *configuration* object that specifies which model to convolve the basis functions with, rather than as a nod
e of the pdf's computation graph.
The `RooAbsAnaConvPdf` builds its own internal basis-function convolutions from it and evaluates *those*.

Until now, the resolution model was nevertheless kept as a (non-value, non-shape) server of the `RooAbsAnaConvPdf`.
As a side effect, importing such a pdf into a `RooWorkspace` also imported the original resolution model as a standalone workspace object, and it leaked into HS3/JSON exports, even though it played no role in the computation.

Starting with ROOT 6.42, a resolution model that is only used as the configuration of a `RooAbsAnaConvPdf` is no longer a server of that pdf, and is therefore not imported into the
workspace on its own anymore. The model remains accessible via `RooAbsAnaConvPdf::getModel()`.

This is not expected to affect typical usage, since the resolution model was never part of the actual likelihood. Workspaces written with older ROOT versions are read back correctly via schema evolution.

## Graphics and GUI

### New POLF and POLN draw options for TH2

Since ROOT 6.36 implementation of "POL" draw option was changed. Angle and radius range automatically scaled to visible histogram range filling full 2*Pi angle and full radius range.
This let display TH2 in polar coordinates for any provided range settings, but produced plots are not intuitive. Therefore two new options for polar coordinates were introduced.

"POLF" - fixed polar coordinates. In such case full histogram X range mapped to -PI .. +Pi.
And Y axis mapped to radius. In case of histogram zooming angle and radius of each bin remains
the same (therefore name "fixed"), just number of displayed bins are reduced.

"POLN" - natural polar coordinates. In this case X axis directly represents angle value in radians (therefore name "natural") and Y axis is just radius.

### Store canvas as HTML file

Now canvas (or several canvases) can be stored in portable HTML file.
Just call `c1->SaveAs("canvas.html")` or invoke correspondent menu item.
To store several canvases in single HTML file one can use:
```cpp
   auto c1 = new TCanvas("c1", "c1", 4);
   auto c2 = new TCanvas("c2", "c2", 4);
   auto c3 = new TCanvas("c3", "c3", 4);
   TCanvas::SaveAll({c1, c2, c3}, "canvases.html");
```
Produced HTML file will include canvas JSON data and JavaScript code to load and display canvas.
Such file can be loaded locally in any web browser or send as attachment in email to colleagues.

### Other graphics improvements

* Both vertical and horizontal `TPaletteAxis` provide now interactive zooming with mouse cursor


## Geometry

### Improved multithreaded `TGeo` navigation

Multithreaded `TGeo` navigation is now faster and more scalable, with improved thread-local state management that avoids
false sharing, releases temporary memory during geometry cleanup, and correctly supports concurrent navigation of
multiple geometries.

For ALICE material-budget lookup-table generation on 28 cores, these changes reduced the runtime from 139 s to 72 s
and improved scaling from 12x to 23x.

The [TGeometry](https://root.cern/doc/master/classTGeometry.html) classes (Geant 3 shapes) have been moved out of Graf3D into their own library.
To link to these classes, use the cmake target `TGeometry` (preferred), `root-config --libs`, or link with `-lTGeometry`.
When ROOT is configured with `-Dgeom=Off`, these classes are now off as well.

The header X3DBuffer.h is no longer part of the installed ROOT headers.

## Documentation and Examples

## Build, Configuration and Testing

### Building the CUDA backend of RooFit separately

RooFit evaluates its models with backend libraries that `libRooBatchCompute`
loads at runtime, one of which, `libRooBatchCompute_CUDA`, is the only part of
ROOT that requires the CUDA toolkit. It is now possible to build that backend
on its own against an already installed ROOT, so that distributions can ship a
CUDA-free ROOT and provide the GPU backend as a separate package:

```bash
cmake -S <root-source-dir>/roofit/batchcompute -B build -DCMAKE_PREFIX_PATH=<root-install-prefix>
cmake --build build
cmake --install build
```

`roofit/batchcompute/CMakeLists.txt` doubles as the top-level `CMakeLists.txt`
of that standalone project, so no CMake code has to be maintained downstream.
By default the library is installed into the library directory of the ROOT
installation it was configured against, which is where RooFit looks for it. See
`roofit/batchcompute/README.md` for the details. Nothing changes for the regular
ROOT build: `-Dcuda=ON` still builds the CUDA backend together with everything
else.

## Built-in packages

**It is strongly recommended to rely on sytem packages, and not built-in ones.**
All the built-in packages are now collected in the builtins directory.
The version of the following packages has been updated:

 - CDT: 2.0.0
 - civetweb: 1.16.git588860
 - bvh: 2.0.gitad9f7e
 - ftgl: 2.4.0
 - giflib: 6.1.3
 - gl2ps: 1.4.2
 - libmathtext: 2016
 - libpng: 1.6.59
 - libjpeg-turbo: 3.2.0
 - libtiff: 4.7.26
 - openssl: 3.5.9
 - openui5: 1.152.0
 - pcre: now pcre2, version 10.47
 - rendercore: 2.3
 - xrootd: 6.2.0
 - xxhash: 0.8.3

## Tracker Items addressed

More than 180 items in ROOT's trackers have been addressed in this release:

* [[6432](https://github.com/root-project/root/issues/6432)] - Build performance does not scale to many cores/threads
* [[6439](https://github.com/root-project/root/issues/6439)] - Failed compilation leaves cling in a "bad state"
* [[6473](https://github.com/root-project/root/issues/6473)] - TTree::Scan truncates leaf names in output
* [[6566](https://github.com/root-project/root/issues/6566)] - TListOf{Functions,DataMembers}::fIds might need thread protection
* [[7123](https://github.com/root-project/root/issues/7123)] - `TClass::Init()` does two lookups where one is enough
* [[7541](https://github.com/root-project/root/issues/7541)] - Cannot catch exceptions on Apple M1
* [[7626](https://github.com/root-project/root/issues/7626)] -  issue with the /dev/stdout redirection into a file
* [[7844](https://github.com/root-project/root/issues/7844)] - TTree::Scan() (and TTree::Draw()) issues with ULong64_t
* [[7968](https://github.com/root-project/root/issues/7968)] - Build option for Address Sanitizer is broken
* [[7979](https://github.com/root-project/root/issues/7979)] - Make TRandom compatible with the std UniformRandomBitGenerator interface
* [[8507](https://github.com/root-project/root/issues/8507)] - [RF] Add documentation for RooKeysPdf, and to its Mirror option
* [[8633](https://github.com/root-project/root/issues/8633)] - Don't replace CMake's find_package
* [[8655](https://github.com/root-project/root/issues/8655)] - [Build system] ROOT installs its builtins directly into `CMAKE_INSTALL_PREFIX`.
* [[9090](https://github.com/root-project/root/issues/9090)] - legacy Makefiles and Configures
* [[9246](https://github.com/root-project/root/issues/9246)] - Enums are not correctly resolved by PyROOT when using an alias within a class/struct
* [[9490](https://github.com/root-project/root/issues/9490)] - [RF] RooMCStudy does not save generated parameters
* [[9758](https://github.com/root-project/root/issues/9758)] - TTreeReaderArray can't read leafs of ULong64_t type (but branches are fine)
* [[10021](https://github.com/root-project/root/issues/10021)] - Add Multithreading support for TGraphErrors Fitting
* [[10240](https://github.com/root-project/root/issues/10240)] - TTree: Creating a branch with std::vector can raise an internal error when mixing I/O systems.
* [[10680](https://github.com/root-project/root/issues/10680)] - [Python] Namespace dropped in std::function cppyy binding
* [[10724](https://github.com/root-project/root/issues/10724)] - PyROOT / cppyy: potentially ineffcicient "member offset" on enumerators
* [[10970](https://github.com/root-project/root/issues/10970)] - [cling] DynamicLibraryManager finds non-exported symbols on macOS
* [[11786](https://github.com/root-project/root/issues/11786)] - TTreePlayer::Scan can't handle left-justified text
* [[12277](https://github.com/root-project/root/issues/12277)] - Duplicate symbols for the JIT
* [[12387](https://github.com/root-project/root/issues/12387)] - [RF] Fitting parameter of interest with Gaussian in RooMCStudy in addition to its pull 
* [[12559](https://github.com/root-project/root/issues/12559)] - Problem to read `TF1`/`TFormula` stored with previous ROOT versions
* [[12701](https://github.com/root-project/root/issues/12701)] - ROOT does not properly recognize if `io_uring` is supported
* [[12738](https://github.com/root-project/root/issues/12738)] - Windows debug builds of root fail. 
* [[12935](https://github.com/root-project/root/issues/12935)] - [RF] Global correlation coefficients after SumW2Error
* [[13185](https://github.com/root-project/root/issues/13185)] - [RF]  RooFit - Pythonic interaction with the RooWorkspace
* [[13287](https://github.com/root-project/root/issues/13287)] - [RF][HF] ShapeFactor with initial shape creates invalid HistFactory XML
* [[14061](https://github.com/root-project/root/issues/14061)] - GitHub Actions don't build exactly the pushed commit
* [[14121](https://github.com/root-project/root/issues/14121)] - gtest-core-metacling-test-TClingLoadUnloadFile fails sporadically
* [[14255](https://github.com/root-project/root/issues/14255)] - [RF] Asymmetry plots with RooFit
* [[15080](https://github.com/root-project/root/issues/15080)] - [DF] Add support and tests for concurrent usage of different RDF objects from different threads
* [[15085](https://github.com/root-project/root/issues/15085)] - PyROOT/cppyy fails  with template functions overload with SFINAE
* [[15323](https://github.com/root-project/root/issues/15323)] - RDF Progress bar seems to ignore Range() calls
* [[15378](https://github.com/root-project/root/issues/15378)] - [PyROOT] Problem with templated binary operator implementations like `operator+`
* [[15818](https://github.com/root-project/root/issues/15818)] - asan segfault due to declaration of function with very long return expression
* [[16210](https://github.com/root-project/root/issues/16210)] - [PyROOT] Automatic downcasting of smart pointers to actual type
* [[16601](https://github.com/root-project/root/issues/16601)] - [6.32/master] Fail to materialise symbols from `libMatrix`
* [[16757](https://github.com/root-project/root/issues/16757)] - [ntuple] Handling of over-aligned types
* [[16765](https://github.com/root-project/root/issues/16765)] - [ntuple] Wrong alignment with transient class member
* [[16852](https://github.com/root-project/root/issues/16852)] - [Python] Regression in accessing protected data members in ROOT 6.32 after cppyy upgrade
* [[16929](https://github.com/root-project/root/issues/16929)] - gtest-tree-ntuple-v7-ntuple-types fails, likely to due fast math.
* [[17057](https://github.com/root-project/root/issues/17057)] - [RDF] support bar charts in RDataFrame
* [[17344](https://github.com/root-project/root/issues/17344)] - [RF] RooFit EvalBackend("cpu") Disables Multi-Core Support During RooMinimizer Minimization
* [[17483](https://github.com/root-project/root/issues/17483)] - wsl2 menu display issue
* [[17713](https://github.com/root-project/root/issues/17713)] - [ntuple] fix I/O rule handling for unversioned classes
* [[18387](https://github.com/root-project/root/issues/18387)] - Additional patch to Clangs CMake code is required in case LLVM was built with `LLVM_LINK_LLVM_DYLIB=ON`
* [[18641](https://github.com/root-project/root/issues/18641)] - [RF][CMake] Make it possible to build RooBatchCompute_CUDA standalone without custom CMake lists file
* [[19122](https://github.com/root-project/root/issues/19122)] - [cppyy] C++11 tests fail on macOS26-Beta+XCode26 Beta1
* [[19208](https://github.com/root-project/root/issues/19208)] - roottest should not pollute source directory with build artifacts
* [[19230](https://github.com/root-project/root/issues/19230)] - [ntuple] Support storage of a SOA using contiguous memory
* [[19358](https://github.com/root-project/root/issues/19358)] - cppyy.gbl.std.function(...) fails for template types like std::vector<int>, ROOT::RVec<int>
* [[19560](https://github.com/root-project/root/issues/19560)] - TTreeIndex fails under Valgrind due to long double emulation
* [[19622](https://github.com/root-project/root/issues/19622)] - RNTuple creates a DAOS container it canont use
* [[19836](https://github.com/root-project/root/issues/19836)] - Potential CMake ordering issue with -Dasan=ON
* [[20213](https://github.com/root-project/root/issues/20213)] - Implement RNTuple redirection from FUSE to XRootD
* [[20245](https://github.com/root-project/root/issues/20245)] - [RooFit] RooResolutionModels create hidden value servers as part of analytical convolutions
* [[20635](https://github.com/root-project/root/issues/20635)] - __cling_Destruct_ symbols strip namespaces and cause errors
* [[20731](https://github.com/root-project/root/issues/20731)] - Investigate/Remove potentially unnecessary explicit linking
* [[20845](https://github.com/root-project/root/issues/20845)] - Crash when using FromSpec with dask and threads
* [[20853](https://github.com/root-project/root/issues/20853)] - [ntuple] add several useful performance metrics
* [[20882](https://github.com/root-project/root/issues/20882)] - [roottest] The stlNoLib test crashes when reading a tree without a dictionary
* [[20907](https://github.com/root-project/root/issues/20907)] - Spurious rebuilds in some circunstances
* [[21005](https://github.com/root-project/root/issues/21005)] - Strange wrapper code generated in cppyy unit test for `cmake_cxxmodules=OFF` with C++17
* [[21071](https://github.com/root-project/root/issues/21071)] - [tmva][sofie] Improve numerical stability and overflow protection for Softplus operator
* [[21203](https://github.com/root-project/root/issues/21203)] - ROOT_GENERATE_DICTIONARY needs to be updated to use DEPFILE instead of IMPLICIT_DEPENDS with add_custom_command
* [[21367](https://github.com/root-project/root/issues/21367)] - Stack corruption in TColorGradient with clang 20 to 22
* [[21371](https://github.com/root-project/root/issues/21371)] - RooFormulaVar @N parameter indices silently shift after workspace serialization
* [[21457](https://github.com/root-project/root/issues/21457)] - ROOT interpreter doesn't handle `<Insert>` key properly at the prompt
* [[21660](https://github.com/root-project/root/issues/21660)] - Memory leak observed when merging root files with histograms
* [[21684](https://github.com/root-project/root/issues/21684)] - [tmva][sofie] Keras parser: Conv2D padding='same' ignores dilation_rate, producing wrong padding
* [[21873](https://github.com/root-project/root/issues/21873)] - [tmva][sofie] Extend Im2col to support asymmetric padding for full ONNX SAME autopad compliance
* [[21978](https://github.com/root-project/root/issues/21978)] - [tmva][sofie] Conv+Add fusion path can drop operator and break downstream Relu type registration
* [[22019](https://github.com/root-project/root/issues/22019)] - [cling] Prompt autocomplete should treat smart pointers as pointers, not as generic classes
* [[22130](https://github.com/root-project/root/issues/22130)] - [ntuple] merging RNTuples produced with ROOT <= 6.34 fails
* [[22157](https://github.com/root-project/root/issues/22157)] - [glcanvas] save as PNG/JPG/BMP image does not work for first time
* [[22164](https://github.com/root-project/root/issues/22164)] - [tree] Heap buffer overflow in TBasket::Streamer when reading TTree baskets with mismatched fNevBufSize and on-disk array length
* [[22214](https://github.com/root-project/root/issues/22214)] - Potential buffer overflow in TFile.cxx
* [[22251](https://github.com/root-project/root/issues/22251)] - Embedded raster images in pdf plotting output
* [[22259](https://github.com/root-project/root/issues/22259)] - Addition of em dash and en dash to TLatex
* [[22284](https://github.com/root-project/root/issues/22284)] - RNTupleDescriptor `RIterator`s are not usable with STL algorithms
* [[22295](https://github.com/root-project/root/issues/22295)] - Expression incorrectly parsed in RDataFrame's Define()
* [[22359](https://github.com/root-project/root/issues/22359)] - "Second registration of" error with a template argument "missing"
* [[22367](https://github.com/root-project/root/issues/22367)] - Wrong interaction of DefinePerSample with downstream varied action
* [[22378](https://github.com/root-project/root/issues/22378)] - SetOSX_SDK.cmake overrides requested `CMAKE_OSX_SYSROOT`
* [[22395](https://github.com/root-project/root/issues/22395)] - TGeoTessellated volumes read from GDML impossible to be closed (6.40.00 regression)
* [[22417](https://github.com/root-project/root/issues/22417)] - 6.40: TBox added to TH1::GetListOfFunctions() appears to be freed before Write
* [[22447](https://github.com/root-project/root/issues/22447)] - 6.40: Glitch in TTree::Draw for 3D (missing bins)
* [[22449](https://github.com/root-project/root/issues/22449)] - [Windows] Macro df002_dataModel.C crashes on exit or exits silently due to potential RVec/Heap issues across DLL boundaries in Cling JIT
* [[22450](https://github.com/root-project/root/issues/22450)] - [Windows] TRootBrowser cannot display non-ASCII filenames (GDK encoding + font fallback)
* [[22464](https://github.com/root-project/root/issues/22464)] - How do we check if a file is corrupted?
* [[22467](https://github.com/root-project/root/issues/22467)] - ACLiC on Fedora 44 and GCC 16 in debug mode triggers uninitialized warning in RVec
* [[22471](https://github.com/root-project/root/issues/22471)] - issue with transparency and interactive mode on macOS
* [[22473](https://github.com/root-project/root/issues/22473)] - [tmva][sofie] Conv generates invalid code and crashes for dilation > 1
* [[22496](https://github.com/root-project/root/issues/22496)] - root -q -e '.gh bug' doesn't work on ssh connections
* [[22501](https://github.com/root-project/root/issues/22501)] - THttpServer/TRootSniffer: exe.json leaks the object returned by the executed method (e.g. TH2::ProjectionY)
* [[22523](https://github.com/root-project/root/issues/22523)] - [tmva][sofie] AveragePool with ceil_mode divides the partial/overhang window by the full kernel size
* [[22541](https://github.com/root-project/root/issues/22541)] - [rfile] Add python documentation
* [[22552](https://github.com/root-project/root/issues/22552)] - [tmva][sofie] Register the existing Swish operator in the ONNX parser
* [[22554](https://github.com/root-project/root/issues/22554)] - [RDataFrame] char type is not recognized in RDF.AsNumpy
* [[22652](https://github.com/root-project/root/issues/22652)] - Wrong behaviour in TTree::[GetMinimum,GetMaximum]
* [[22727](https://github.com/root-project/root/issues/22727)] - [jsroot] Scroll breaks on SWAN when using %rootbrowse
* [[22740](https://github.com/root-project/root/issues/22740)] - [RF] Unexpected behaviour when clipping domain ranges and fitting
* [[22755](https://github.com/root-project/root/issues/22755)] - TTreeFormula: sqrt() evaluates sqrt(abs(x)) for negative arguments
* [[22769](https://github.com/root-project/root/issues/22769)] - Histogram with fit function not correctly drawn to canvas in Jupyter notebook with JsROOT on
* [[22781](https://github.com/root-project/root/issues/22781)] - Several memory leaks in TLegend and TLatex
* [[22831](https://github.com/root-project/root/issues/22831)] - TGDMLParse silently drops elements whose name collides with a material name after `0x` pointer-suffix stripping
* [[22843](https://github.com/root-project/root/issues/22843)] - TMath::Laplacian computes wrong finite differences
* [[22855](https://github.com/root-project/root/issues/22855)] - [ntuple] double free when resizing small RVec
* [[22856](https://github.com/root-project/root/issues/22856)] - [ntuple] REntry::BindRawPtr() does not type-check
* [[22858](https://github.com/root-project/root/issues/22858)] - [RF] RooBinIntegreator gives wrong result for integrals that cut thought bin boundaries
* [[22938](https://github.com/root-project/root/issues/22938)] - `R__memcompress` fails with limited stack size
* [[22953](https://github.com/root-project/root/issues/22953)] - [macOS/X11] Rebuilding and reusing TGPopupMenu makes context menus block in XGetGeometry
* [[22954](https://github.com/root-project/root/issues/22954)] - [DF] Support for RHist with VariationsFor
* [[22959](https://github.com/root-project/root/issues/22959)] - [RF] Bug with RooExtendPdf produces inconsistency in rf204b_extendedLikelihood_rangedFit.C between before 6.26 and after 6.28
* [[22981](https://github.com/root-project/root/issues/22981)] - [ntuple] streamer field end up using wrong type name
* [[23014](https://github.com/root-project/root/issues/23014)] - TASImage::DrawCircle() fails with semi-transparent color
* [[23026](https://github.com/root-project/root/issues/23026)] - ROOT fails to configure with CMake 4.4
* [[23036](https://github.com/root-project/root/issues/23036)] - system.rootrc ships three settings that nothing reads (WebGui.HttpLoopback, WebGui.WaitForTmout, Browser.StartUrl)
* [[23081](https://github.com/root-project/root/issues/23081)] - [TMVA][SOFIE] Severe Memory Bloat in ROperator_Pool due to Unused _xpad Vector Allocatio
* [[23114](https://github.com/root-project/root/issues/23114)] - [CI] publish_binaries maybe using an unset variable.
* [[23120](https://github.com/root-project/root/issues/23120)] - `TASImage::DrawEllips2()` works incorrectly when filling tilted ellipse
* [[23125](https://github.com/root-project/root/issues/23125)] - Histograms containing NaN can't be drawn.
* [[23126](https://github.com/root-project/root/issues/23126)] - New macOS SDK detection mechanism breaks under certain circumstances LCG Builds on macOS15
* [[23146](https://github.com/root-project/root/issues/23146)] - RNTuple: I/O rule with multiple source members assigns every target the first source value of its type
* [[23148](https://github.com/root-project/root/issues/23148)] - `TASImage::DrawEllips2`: Flood fill leaks through contour gap in asim_straight_ellips() (affects up to 64% of axis-aligned non-circular ellipses)
* [[23152](https://github.com/root-project/root/issues/23152)] - [clingutils] build regression between 6.40 and 6.42 with builtin_llvm=OFF
* [[23155](https://github.com/root-project/root/issues/23155)] - [clang] DISABLE_LLVM_LINK_LLVM_DYLIB no longer honored after upgrade to LLVM22 when configuring with builtin_llvm=OFF
* [[23186](https://github.com/root-project/root/issues/23186)] - [tmva][sofie] Range operator emits non-compiling size expression for symbolic INT64 inputs
* [[23192](https://github.com/root-project/root/issues/23192)] - TGeo: stepping does not resolve a MANY overlap when the overlapping node has an ONLY daughter
* [[23194](https://github.com/root-project/root/issues/23194)] - [TCling] Assertion in Sema::CheckTypenameType failing (possible regression after LLVM22 import)
* [[23221](https://github.com/root-project/root/issues/23221)] - [RF][HS3] RooJSONFactoryWSTool silently exports an empty document after `pdf.getVal(normSet)` or `createIntegral`
* [[23237](https://github.com/root-project/root/issues/23237)] - [geom] GDML round trip loses reflections: on export, and on import inside assemblies
* [[23275](https://github.com/root-project/root/issues/23275)] - Add support for missing ONNX inverse hyperbolic operators (Asinh, Acosh, Atanh) in TMVA SOFIE
* [[23308](https://github.com/root-project/root/issues/23308)] - [sofie] ONNX GNN model fails to convert: missing operators, dynamic-shape gaps, and two codegen bugs
* [[23342](https://github.com/root-project/root/issues/23342)] - [RF] Nested `RooSimultaneous` double counts extended sub-pdfs
* [[23374](https://github.com/root-project/root/issues/23374)] - Typo in CI replication message
* [[23378](https://github.com/root-project/root/issues/23378)] - [interpreter] wrong CLANG_CMAKE_DIR when builtin_llvm=OFF but builtin_clang=ON
* [[23402](https://github.com/root-project/root/issues/23402)] - Enable `http=ON` on Ubuntu 26.10
* [[23417](https://github.com/root-project/root/issues/23417)] - RDataLoader hangs with batches_in_memory=1
* [[23444](https://github.com/root-project/root/issues/23444)] - [RF] Issue of of multi-range fit in RooSimultaneous
* [[23450](https://github.com/root-project/root/issues/23450)] - `TMath::IsInside()` returns wrong results when Int_t is used
* [[23459](https://github.com/root-project/root/issues/23459)] - PyROOT crashes when IPython formats an empty std::optional
* [[23491](https://github.com/root-project/root/issues/23491)] - `TGraphErrors::Draw("3")` changes the error band when the axis range is changed
* [[23493](https://github.com/root-project/root/issues/23493)] - [cling] Clang is conditionally invoked with the wrong driver mode depending on the executable's name
* [[23500](https://github.com/root-project/root/issues/23500)] - [tmva][sofie] 
* [[23502](https://github.com/root-project/root/issues/23502)] - [tmva][sofie] BatchNormalization ignores the epsilon attribute (hardcoded 1e-5)
* [[23512](https://github.com/root-project/root/issues/23512)] - [RF] Wrong expected events for extended fits in multiple disjoint ranges with the new evaluation backend
* [[23523](https://github.com/root-project/root/issues/23523)] - [tmva][sofie] Where computes the wrong output shape when its inputs broadcast
* [[23547](https://github.com/root-project/root/issues/23547)] - [tmva][sofie] LogSoftmax returns -inf when the inputs are far apart
* [[23559](https://github.com/root-project/root/issues/23559)] - bothersome excel file names error in console when using TBrowser
* [[23569](https://github.com/root-project/root/issues/23569)] - TTree::SetBranchAddress does not find friend branch
* [[23582](https://github.com/root-project/root/issues/23582)] - roottest residual Makefiles
* [[23628](https://github.com/root-project/root/issues/23628)] - regression in h2root
* [[23648](https://github.com/root-project/root/issues/23648)] - root CLI no longer opens files before executing -e commands
* [[23655](https://github.com/root-project/root/issues/23655)] - [tmva][sofie] Inputs of unsupported type (e.g. float16) give code that does not compile instead of an error
* [[23664](https://github.com/root-project/root/issues/23664)] - [Windows] `<filesystem>` doesn't work in interpreted code
* [[23681](https://github.com/root-project/root/issues/23681)] - Quadratic performance in TH1Merger::LabelMerge
* [[23706](https://github.com/root-project/root/issues/23706)] - Buffers of 2^24 bytes or slightly more are written uncompressed (RNTuple pages and TTree baskets)
* [[23712](https://github.com/root-project/root/issues/23712)] - rootcling crashes (SIGSEGV) on Xcode 27 when SDKROOT is set to an empty string
* [[23716](https://github.com/root-project/root/issues/23716)] - [hist] Merging 3D histograms with alphanumeric labels on the axes is broken
* [[23724](https://github.com/root-project/root/issues/23724)] - RooExponential: Issue with codegen gradient near zero slope
* [[23734](https://github.com/root-project/root/issues/23734)] - [tmva][sofie] Selu ignores its alpha and gamma attributes
* [[23767](https://github.com/root-project/root/issues/23767)] - out of bound access when building TLegend
* [[23772](https://github.com/root-project/root/issues/23772)] - [meta] TDataMember::GetOptions() return wrong values for enum member
* [[23790](https://github.com/root-project/root/issues/23790)] - [RDataFrame] Code that transforms `#var` alias to `R_rdf_sizeof_var` does not respect string literals
* [[23842](https://github.com/root-project/root/issues/23842)] - RDataLoader: wrong train/validation split with lazy loading, filtered inputs and shuffle=False
* [[23866](https://github.com/root-project/root/issues/23866)] - stressLinear.cxx cannot be interpreted in 6.40.04
* [[ROOT-6035](https://its.cern.ch/jira/browse/6035)] - Counter-intuitive behaviour of RooExtendPdf in fitTo()
* [[ROOT-6203](https://its.cern.ch/jira/browse/6203)] - Same name classes in different namespaces
* [[ROOT-8000](https://its.cern.ch/jira/browse/8000)] - TTree::Draw evaluating "innocent" files in the current working directory
* [[ROOT-9302](https://its.cern.ch/jira/browse/9302)] - TClingCallbacks::InclusionDirective() re-includes the header for fun
* [[ROOT-9562](https://its.cern.ch/jira/browse/9562)] - Support UTF-8 at prompt
* [[ROOT-10069](https://its.cern.ch/jira/browse/10069)] - AsymptoticCalculator.GenerateAsimovData cannot generate dataset when more than one variable is floating
* [[ROOT-10453](https://its.cern.ch/jira/browse/10453)] - Poor design of the RooIntegralMorph class
* [[ROOT-10589](https://its.cern.ch/jira/browse/10589)] - Exception silently caught by Jupyter ROOT kernel
* [[ROOT-10707](https://its.cern.ch/jira/browse/10707)] - cling ptr check suppression vs decl extraction broken
* [[ROOT-10825](https://its.cern.ch/jira/browse/10825)] - RooHistPdf gets crazy in presence of negative weights while it is supposed to manage them and ignore them
