% ROOT Version 6.06 Release Notes
% 2015-12-08
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.06/00 is scheduled for release in November, 2015.

For more information, see:

[http://root.cern.ch](http://root.cern.ch)

The following people have contributed to this new version:

 David Abdurachmanov, CERN, CMS,\
 Bertrand Bellenot, CERN/SFT,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, FNAL,\
 Cristina Cristescu, CERN/SFT,\
 Olivier Couet, CERN/SFT,\
 Kyle Cranmer, NYU, RooStats,\
 Gerri Ganis, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Enrico Guiraud, CERN/SFT, \
 Burt Holzman, Fermilab, CMS,\
 Lukasz Janyst, CERN/IT,\
 Christopher Jones, Fermilab, CMS,\
 Wim Lavrijsen, LBNL, PyRoot,\
 Sergey Linev, GSI, http, JSROOT, \
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Axel Naumann, CERN/SFT,\
 Danilo Piparo, CERN/SFT,\
 Timur Pocheptsov, CERN/SFT,\
 Fons Rademakers, CERN/IT/Openlab,\
 Enric Tejedor Saavedra, CERN/SFT,\
 Liza Sakellari, CERN/SFT,\
 Manuel Tobias Schiller,CERN, LHCb\
 David Smith, CERN/IT,\
 Matevz Tadel, UCSD/CMS, Eve, \
 Vassil Vassilev, CERN/SFT \
 Wouter Verkerke, NIKHEF/Atlas, RooFit, \
 Omar, Zapata, Medellin, Columbia \
 Maciej Zimnoch, GSoC, Poland

## ROOT reference manual

The ROOT reference manual has been moved into Doxygen. Still some work and
polish has to be done but the reference guide in this new format is now online
and can be seen from the [ROOT home page](https://root.cern.ch/doc/master/index.html).

## Core Libraries

### Dictionary generation

Fixed the dictionary generation in the case of class inside a namespace
marked inlined.

Added mechanisms to stop the dictionary generation while parsing the XML and while selecting in presence of duplicates.

Fix [ROOT-7760] : fully allow the usage of the dylib extension on OSx.

Fix [ROOT-7723] : allow IOCtors to have as argument a ref to a type called __void__.

We added a dictionary for map<string,string> as part of the default STL dictionary.

We added support for template parameter packs in class name involved in the I/O.

### Thread safety and thread awareness

We added the function `TMethodCall::GetCallFunc` to allow direct access to the function wrapper.

We reduced thread serialization in `TClass::GetCheckSum`, `TClass::GetBaseClassOffset` and `TClass::Property`

`TObjArray::Delete` was updated to allow its caller to explicitly avoid costly checks (extra RecursiveRemove and lock)

We removed the need to create a TThread object per thread in a multi-threaded application. Now ROOT can be used with any threading model (e.g. OpenMP, STL threads, TBB) transparently.  All the internal synchronisation mechanisms of ROOT are activated by a single call: `ROOT::EnableThreadSafety()` which is the successor of the existing `TThread::Initialize`.  This call must take place if ROOT needs to be used in a thread safe manner.

The implementation of TSemaphore was redone based on C++11 thread primitive in order to prevent cases where some of request post were lost.

### TDirectory::TContext

We added a default constructor to `TDirectory::TContext` which record the current directory
and will restore it at destruction time and does not change the current directory.

The constructor for `TDirectory::TContext` that takes a single TDirectory pointer as
an argument was changed to set `gDirectory` to zero when being passed a null pointer;
previously it was interpreting a null pointer as a request to *not* change the current
directory - this behavior is now implement by the default constructor.

### Collections

In THashList and THashTable, GetListForObject now returns a pointer to const as modifying the returned list (in particular adding to it) can break invariant of THashTable so we need to clearly mark the list as not being allowed to be modified.

In TSeqCollection::Merge, we no longer delete the object in the case where the original collection is marked as a owner.

We resolved a memory leakage occuring as a consequence of repeated calls to `TClonesArray::AbsorbObjects` and `TClonesArray::Clear` [ROOT-6996].  A similar problem was affecting `TClonesArray::operator=`, `TClonesArray::Expand` and `TClonesArray::ExpandCreate` and was also solved.  `TClonesArray` reliance on global state during the destruction of the elements was decreased (removing use of `TObject::SetDtorOnly`)

### Global resources.

Several tweaks to if and when, resources held by the global ROOT object (TROOT, TApplication) are deleted.  When the default TApplication is replaced by a user provide TApplication, do not call EndOfProcessCleanups and co. and thus do not delete TFiles, TSockets or TColors that have already been created.  In EndOfProcessCleanups, we now delete the objects held in TROOT's TDirectory part.  If the libCling library is unloaded, this now induces an immediate tear down of the ROOT resources; consequently objects might be deleted sooner in the process tear down process on some platforms.

TObject instances allocated as part of an array and made part of a collection, as for example the TCanvas instances into the global list of instances, are not longer deleted if the content of the collection is deleted.  Technically the element of the array are now treated by collections as if they have been allocated on the stack.  This fixes the issue described at [ROOT-7846].

### Code Cleanups.

Several definition where moved from the global or ROOT namespace to the ROOT::Internal namespace as they are not intended to be used outside of ROOT, including: `gROOTLocal` and related functions, `TSchemaHelper`, `TSchemaMatch`, `TSchemaType`, `RStl`, `ROOT::TROOTAllocator`, `TSchemaRuleProcessor`, `TStdBitsetHelper`, `TInitBehavior`, `TDefaultInitBehavior`, `DefineBehavior`, `THnBaseBrowsable`, `THnBaseBinIter`, `GenericShowMembers`, `TOperatorNewHelper` and `BranchProxy` implementations classes.

Several definition where moved from the global or ROOT namespace to the ROOT::Details namespace as they are intended to be used in 'expert' level code and have a lower level of backward compatibility requirement.  This includes `TCollectionProxyInfo`, `TSchemaRuleSet`.

## Interpreter

ROOT can now dump the context of STL collections, for instance `map<string,int>`. A few ROOT types print their content, too.

Fixed the handling of the current directory in `#include` of system headers, avoid problem with local files named `new` or `vector`.

Fixed the issue with the ROOT special variable where the objects were read from the file at each and every access by caching those object.  See [ROOT-7830] for example.

This release contains several bug fixes and improvements, notably in unloading and performance.

> NOTE: The GCC 5 ABI is *not* supported yet, due to a lack of support in clang.

## I/O Libraries

### hadd

We extended the `hadd` options to allow more control on the compression settings use for the
output file.  In particular the new option -fk allows for a copy of the input
files with no decompressions/recompression of the TTree baskets even if they
do not match the requested compression setting.

New options:

- `-ff` allows to force the compression setting to match the one from the first input
- `-fk[0-209]` allows to keep all the basket compressed as is and to compress the meta data with the given compression setting or the compression setting of the first input file.
- `-a` option append to existing file
- The verbosity level is now optional after -v

### Command line utilities

We added command line utilities to streamline very common operations performed on root files, like listing their content or creating directories.
The command line utilities are:
- `rootbrowse`: to open the file in a TBrowser
- `rootcp`: to copy content from one file to another
- `rooteventselector`: to select a subset of the events in a tree contained in a file
- `rootls`: to list the content of a rootfile
- `rootmkdir`: to create a directory in a rootfile
- `rootmv`: to move content across files
- `rootprint`: to plot content (histograms, graphs) of files
- `rootrm`: to remove content from files
These utilities took inspiration from the well known *nix commands and all offer the `-h` switch which provides documentation for all options available and example invocation lines.

### TBufferFile.

We updated TBuffer::Expand to properly shrink the buffer when requested, hence reducing memory usage in some cases.

### I/O New functionalities

We added support for template parameter packs in class name involved in the I/O.


## TTree Libraries

### Improvement of handling of default number of entries

A new const expression value: `TTree::kMaxEntries` has been introduced to
express the largest possible entry number in a `TTree`.  This is used in
two main cases:

- as the default value for the requested number of entries a routine should be
applied to; for example this is used for `TTree::Draw` and `TTree::Process`.
Previously the default was only 1 billions entries, causing those routines to
end early in case of very large trees.

- as the default value for the number of entries returned by TChain::GetEntriesFast.
The previous value was kBigNumber (set to 1234567890) and internally (but somewhat
inconsistently, see [ROOT-6885]) a larger value was used (named theBigNumber).  Now
`TTree::kMaxEntries` is used throughout TChain.

`TChain::kBigNumber` is deprecated and its value has been changed to be equal
to `TTree::kMaxEntries`.

### MakeSelector

`TTree::MakeSelector` has been update to generate a code skeleton based on the `TTreeReader` rather than the old style relying on numeric data members replacements for the user objects.  The main advantage is the lifting of the problem related to the fact that the old style was using fixed size array to represent variable size collection.

`TTree::MakeSelector` takes an option parameter that can be used to specify the branches that will have a data member.
- If option is `"=legacy"`, a pre-ROOT6 selector will be generated (data members and branch pointers instead of TTreeReaders).
- If option is empty, readers will be generated for each leaf.
- If option is "@", readers will be generated for the topmost branches.
- Individual branches can also be picked by their name:
  - "X" generates readers for leaves of X.
  - "@X" generates a reader for X as a whole.
  - "@X;Y" generates a reader for X as a whole and also readers for the
    leaves of Y.
  - For further examples see the figure below.

\image html ttree_makeselector_option_examples.png

The generated code in selector.h includes the following:
- Identification of the original Tree and Input file name
- Definition of selector class (data and functions)
- The following class functions:
  - constructor and destructor
  - void    Begin(TTree *tree)
  - void    SlaveBegin(TTree *tree)
  - void    Init(TTree *tree)
  - Bool_t  Notify()
  - Bool_t  Process(Long64_t entry)
  - void    Terminate()
  - void    SlaveTerminate()

The class selector derives from TSelector.
The generated code in selector.C includes empty functions defined above.

To use this function:

- connect your Tree file (eg: `TFile f("myfile.root");`)
- `T->MakeSelector("myselect");`

where T is the name of the Tree in file myfile.root
and myselect.h, myselect.C the name of the files created by this function.
In a ROOT session, you can do:
``` {.cpp}
  root > T->Process("myselect.C")
```

### Other improvements.

We fixed the handling of the case when an object, stored in a TTree, used to have a defaulted Streamer and when the TTree is being read, the object now has a custom Streamer that we must use.

## Histogram Libraries

### TH1

* Fix a bug in using the buffer with weights different than one
* Remove the `kCanRebin` bit, that it was not used anymore. Its functionality is replaced by the `TH1::SetCanExtend` function.



### TGraph

* `TGraph::GetHistogram()` was resetting the TimeDisplay attribute of axis.
The problem was reported [here](https://sft.its.cern.ch/jira/browse/ROOT-7766).
* Change `TGraph::ComputeRange`: in case of log scale the minimum along X and
Y axis are now set to the lowest positive values of the graph. Previously a % of the
maximum was used which may hide some points like in the following example
``` {.cpp}
{
   TGraph * gr = new TGraph(10);
   for (int i = 0;i<10;i++) gr->SetPoint(i,i,TMath::Exp(-10.0*i));
   for (int i = 5;i<10;i++) gr->SetPoint(i,i,0.);
   gr->Draw("apl");
   gr->SetMarkerStyle(20);
   gPad->SetLogy(true);
}
```
The problem was reported [here](https://root.cern.ch/phpBB3/viewtopic.php?f=3&t=20484).


### TGraph2D

Add a new implementation for Delauney interpolation using the triangle code from Jonathan Shewchuk, see [[ http://www.cs.cmu.edu/~quake/triangle.html ]]. 
A new class for Delauney triangulator and interpolation has been added in the MathCore library  ( `ROOT::Math::Delauney2D` ).


### Fitting

* Improve thread safety of TH1::Fit by making static member of TVirtualFitter and TMinuitMinimizer thread local.  This fixes [ROOT-7791].
* Fix some bugs in TF1NormSum (to fit normalized sum of functions) and in TF1Convolution
* Add a new histogram fitting option, `WIDTH` to fit directly density. The bin content in this case is scaled by the histogram bin width

### TFormula

* Fix several bugs in the new TFormula class.
* Add as new pre-defined functions: `crystalball`, `breitwigner` and `cheb0,cheb1,...cheb10` for the Chebyshev polynomials. 

## Math Libraries

### Random numbers

* Move from MathMore to MathCore the class `ROOT::Math::Random`. Make it a new interface class for random number generation. Add interfaces for standard
ROOT random engines, GSL random engines and random engines provided by the C++ standard library (`std::random`).
* Add a new randomengine, `MIXMAX` based on matrix-recursive random number generator from Kostas and George Savvidy. See this [paper](http://dx.doi.org/10.1016/j.cpc.2015.06.003).

## R Interface

Apply several improvements in the interface to R, allowing to use R functions within ROOT.
See more at the [ROOT-R User Guide](http://oproject.org/tiki-index.php?page=ROOT%20R%20Users%20Guide). 

## TMVA

Add new TMVA plug-in based on R and Python (using Scikit-Learn) 
* See the [RMVA Web page](http://oproject.org/tiki-index.php?page=RMVA) for a detailed description of the new TMVA method based on R
* See the [PyMVA Web page](http://oproject.org/tiki-index.php?page=PyMVA) for detailed description of the machine learning methods added in TMVA and based on the Python Scikit-Learn package.

## RooFit Libraries
  

## 2D Graphics Libraries

### THistPainter

Improve the algorithm to compute the lower limit of an axis in log scale when its
real lower limit is 0. The problem was reported in ROOT-7414.

Using the `COL` option with histograms having some negative bins; the empty bins
(containing 0) are drawn. In some cases one wants to not draw empty bins
(containing 0) of histograms having a negative minimum. The option `1`, used with
the option `COL`, allows to do that.

Implement the Log option for `CANDLE` plots as requested
[here](https://root.cern.ch/phpBB3/viewtopic.php?f=3&t=20225&p=87006#p87006).

### TTeXDump

From Dmitry Kalinkin (via github): Fix file corruption in `TTeXDump::DrawPolyMarker`
The current implementation of `TTeXDump` uses `TVirtualPS::PrintFast` based methods
to output TeX markup with automatic line-wraps. Yet these methods are optimized for
PostScript format where there are a lot of space characters that are used for newline
placement. Current `TTeXDump::DrawPolyMarker` would often produce a long contiguous lines
that trigger a forceful linewrap that can happen in the middle of real number constant
(ignored by latex) or even in the middle of latex command (producing incorrect file).
One solution would be to rewrite TTeXDump using only `PrintRaw` (that you can't mix
with `PrintStr/PrintFast/WriteReal`). The other would be to fix `PrintFast` to not
introduce forced newline. The third option is less intrusive and just adds additional
spaces to provide clues for the proper line wrapping (this is the one implemented in
this change).

### TLatex

Make sure the line width used to draw `#sqrt` is always >= 1.

When a global text alignment was set the `TLatex`characters `#minus`, `#plus`,
`#mp`, `#hbar`, and `#backslash` were mis-aligned. The following macro demonstrate
the problem:

``` {.cpp}
{
   gStyle->SetTextAlign(22);
   TLatex t(.5,.5,"#minus100 #mp100 #plus100 #hbar #backslash");
   t.Draw();
}
```

The angle of a `TLatex` object was set to 0 if the `GetYsize` method was called.

### TColor

New palette `kViridis`. It was presented at SciPy2015 by Stéfan van der Walt and
Nathaniel Smith. It is now matplotlib's current default color map.

![Viridis](palette_112.png)


### TMultiGraph

Ignore empty graphs when computing the multi-graph range at painting time.

### TASImage

A left click on a image produced a one pixel zoom.

### TCreatePrimitives

The ending of a polyline creation is based on the closeness of the two last
entered points. The previous algorithm was based on user coordinates. It is now
based on pixel to avoid the problem reported
[here](https://root.cern.ch/phpBB3/viewtopic.php?f=3&t=20343).

### TCanvas

When the first canvas created by ROOT was in batch mode, it was note possible to
come back in interactive mode for the next canvases. this problem was reported
[here](https://root.cern.ch/phpBB3/viewtopic.php?f=3&t=20354).

### Cocoa Backend

Sometimes the mouse cursor did not change back to the window manager arrow when
exiting a `TCanvas`.

### `freetype` library

Updates `builtin_freetype` to 2.6.1 (current upstream version), which can detect
`PPC64LE` machine. This was compiled and tested on `SLC6 + ICC + x86_64`,
`F21 + GCC + ppc64le`, `MacOSX 10.11.1 + Xcode 7.1` and `Windows (ROOT 5.34)`.
`$ROOTSYS/graf2d/freetype/src/README` was removed, because no issues were noticed
with `ICC` compiler and `-Wall -pedantic -ansi` flags.
Additionally `--with-png=no --with-bzip2=no` flags are passed to freetype
configuration script. Default values for these options are auto.
`freetype` finds `libpng` and `libbzip2` on the system and builds extra
modules. Then attempting to link against `freetype` one would need to link
`-lpng -lbzip2` explicitly otherwise linking will returns in undefined
references. Otherwise we would need to check for `libpng` and `libbzip2` on the system
and adjust `FREETYPE_LIBRARIES` to include `-lpng` and `-lbzip2`.
The current solution goes for the minimal configuration. The original request for
this update was posted [here](https://sft.its.cern.ch/jira/browse/ROOT-7631).

## 3D Graphics Libraries


## Geometry Libraries


## Database Libraries


## Networking Libraries

### THttpServer

Support of POST HTTP requests. For example, ROOT objects can be send with POST request and used as arguments of
objects method execution in exe.bin and exe.json requests. Request and response HTTP headers are now directly accessible in THttpCallArg class

When command is registered with THttpServer::RegisterCommand() method,
one could configure additional arguments which should be submitted when
command is executed with cmd.json requests

Introduce restriction rules for objects access with THttpServer::Restrict() method.
Up to now general read-only flag was applied - either
everything read-only or everything is fully accessible.
Now one could restrict access to different parts of
objects hierarchy or even fully 'hide' them from the client.
Restriction based on user account name, which is applied
when htdigest authentication is configured.
One also able to allow execution of selected methods.

Implement multi.bin and multi.json requests.
One could request many items with single HTTP request.
Let optimize communication between server and client.

With *SNIFF* tag in ClassDef() comments one could expose different properties,
which than exposed by the TRootSniffer to the client with h.json requests.
Such possibility ease implementation of client-side code for custom classes.

Allow to bind http port with loopback address.
This restrict access to http server only from localhost.
One could either specify 'loopback' option in constructor:
    new THttpServer("http:8080?loopback")
or in clear text specify IP address to which http socket should be bind:
    new THttpServer("http:127.0.0.1:8080")
If host has several network interfaces, one could select one for binding:
    new THttpServer("http:192.168.1.17:8080")

### TNetXNGFileStager
Fixed ROOT-7703. This restores the behavior of Locate() to that found with
TXNetFileStager: Rather than return only the xrootd server's reply, the endpoint
hostname is looked up and Locate() returns the full url, including the path.

### TWebFile
Fixed ROOT-7809. Returns an error for a redirect which does not specify the new
URI, rather than going into a loop.

Fixed ROOT-7817. Avoid a crash under some circumstances when trying to open an
invalid path.

## GUI Libraries


## Montecarlo Libraries


## Multi-processing

With this version we introduce a new module, core/multiproc, for multi-processing on multi-core machines. This module is based on fork technology and offers an interface inspired from Python multiprocessor module. The new interface, implemented in the class TProcPool, provides the possibility to perform in parallel a very generic set of tasks, described by macros, functions or lambdas.

This illustrates the usage of lambdas:

``` {.cpp}
{
  TProcPool pool;
  auto ten = pool.MapReduce([]() { return 1; }, 10, [](std::vector<int> v) { return std::accumulate(v.begin(), v.end(), 0); })
}
```


And this how it can be used to generate ten histos and merge them:

``` {.cpp}
{
TObject *CreateAndFillHists()
{

  TH1F *h = new TH1F("h", "", 100, -3., 3.);
  h->SetDirectory(0);
  h->FillRandom("gaus", 1000);
    return h;
}

    TProcPool pool;
    auto hist = pool.MapReduce(CreateAndFillHists, 10, PoolUtils::ReduceObjects);
    hist->DrawClone();
}
```

Tutorials illustrating other usages of the new class TProcPool are available under tutorials/multicore.


## Language Bindings

### Notebooks
We provided integration of ROOT with the Jupyter technology, integrating ROOT with Python Notebooks and providing a ROOT Kernel like functionality - de facto an enhanced C++ web based shell. Tab completion, output and graphics inlining have been added. These functionalities are automatically available upon import of the ROOT module in a Notebook or at startup of a ROOT prompt kernel.
We made it easier to use ROOT notebooks locally, by providing a 'root --notebook' command option to start a local notebook server customised with all the ROOT features.

New tutorials and code examples have been provided. The simplest example showing the integration of ROOT with the notebook technology can be found [here](https://root.cern.ch/notebooks/HowTos/HowTo_ROOT-Notebooks.html) and many more snippets [here](https://root.cern.ch/code-examples#notebooks).

Support for capturing large outputs (stderr/stdout) coming from C++ libraries has been added.

## JavaScript ROOT

- support registered in THttpServer commands with arguments.
- provide workaround for websites using require.js and older jquery-ui
- support custom requests to remote objects, demonstrated in httptextlog.C tutorial
- rewrite draw.htm (page for individual object drawing) to support all custom features as main gui does
- See also the [JSRoot 3.9 examples page](https://root.cern.ch/js/3.9/) and the [JSRoot 3.9 release notes](https://github.com/linev/jsroot/releases/tag/3.9)

## Class Reference Guide

The ROOT [reference guide](https://root.cern.ch/doc/master/index.html) is moving
to the Doxygen system. Doxygen is the de-facto standard for code documentation. It offers
many nice features we are now exploring and trying to get the best of them. Having
[MathJax rendered math formula](https://root.cern.ch/doc/master/classTSpectrum2.html#a482a7f144b9cc1b0405d0ac0d8cc9bbb)
is one of them. The documentation can be structured in a more logical way
[using groups](https://root.cern.ch/doc/master/modules.html). Still there is a lot
to do but big progresses have been done. We developed also a Doxygen filter allowing
to execute macros given as examples in the documentation and show the
[resulting picture directly in the documentation](https://root.cern.ch/doc/master/classTHistPainter.html#HP16).

## Tutorials

[The tutorials](https://root.cern.ch/doc/master/group__Tutorials.html) in
`$ROOTSYS/tutorials` are also presented on the web thanks to Doxygen. They
are now part of the reference guide allowing nice cross-referencing with the
classes documentation. Here also a filter has been developed to generate
[the resulting picture](https://root.cern.ch/doc/master/ContourList_8C.html).

## Build, Configuration and Testing Infrastructure

ROOT uses the CMake cross-platform build-generator tool as a primary build system. CMake does not build the project, it generates the files needed by your build tool (GNU make, Ninja, Visual Studio, etc) for building ROOT. The classic build with configure/make is is still available but it will not be evolving with the new features of ROOT.

We added full support for C++14.

Minor changes in the build system:

- Renamed build option POSTGRESQL_LIBRARIES to POSTGRESQL_LIBRARY
- Added build option `builtin_openssl` to build OpenSSL internally. This is specially needed for the latest Mac OSX (El Capitan)
