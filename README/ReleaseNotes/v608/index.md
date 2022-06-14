% ROOT Version 6.08 Release Notes
% 2015-11-12
<a name="TopOfPage"></a>

## Introduction

ROOT version 6.08/00 is scheduled for release in October, 2016.

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 David Abdurachmanov, CERN, CMS,\
 Adrian Bevan, Queen Mary University of London, ATLAS,\
 Attila Bagoly, GSoC,\
 Bertrand Bellenot, CERN/SFT,\
 Rene Brun, CERN/SFT,\
 Philippe Canal, Fermilab,\
 Andrew Carnes, University of Florida, CMS, \
 Cristina Cristescu, CERN/SFT,\
 Olivier Couet, CERN/SFT,\
 Gerri Ganis, CERN/SFT,\
 Paul Gessinger, CERN/SFT,\
 Andrei Gheata, CERN/SFT,\
 Luca Giommi, CERN/SFT,\
 Sergei Gleyzer, University of Florida, CMS\
 Christopher Jones, Fermilab, CMS,\
 Wim Lavrijsen, LBNL, PyRoot,\
 Sergey Linev, GSI, http,\
 Pere Mato, CERN/SFT,\
 Lorenzo Moneta, CERN/SFT,\
 Abhinav Moudgil, GSoC, \
 Axel Naumann, CERN/SFT,\
 Simon Pfreundschuh, GSoC, CERN/SFT,\
 Danilo Piparo, CERN/SFT,\
 Timur Pocheptsov, CERN/SFT,\
 Fons Rademakers, CERN/IT,\
 Paul Russo, Fermilab,\
 Enric Tejedor Saavedra, CERN/SFT,\
 George Troska, Dortmund Univ.,\
 Liza Sakellari, CERN/SFT,\
 Alex Saperstein, ANL,\
 Manuel Tobias Schiller, CERN/LHCb,\
 David Smith, CERN/IT,\
 Peter Speckmayer,\
 Tom Stevenson, Queen Mary University of London, ATLAS\
 Matevz Tadel, UCSD/CMS, Eve,\
 Peter van Gemmeren, ANL, ATLAS,\
 Xavier Valls, CERN/SFT, \
 Vassil Vassilev, Fermilab/CMS,\
 Stefan Wunsch, KIT, CMS\
 Omar Zapata, University of Antioquia, CERN/SFT.
 

<a name="core-libs"></a>

## General

* Remove many instances of new warnings issued by gcc 6.1
* Significant update of the valgrind suppression file to hide intentional lack
of delete of some entities at the end of the process.
* Resolved several memory leaks.
* Added deprecation system: when compiling against interfaces marked R__DEPRECATED, the compiler will issue a warning showing the ROOT version when the interface will be removed.
* From this version on, building ROOT with CMake requires CMake &gt;= 3.4.3

## Core Libraries

ROOT prepares for [cxx modules](http://clang.llvm.org/docs/Modules.html). One of
the first requirements is its header files to be self-contained (section "Missing
Includes"). ROOT header files were cleaned up from extra includes and the missing
includes were added.

This could be considered as backward incompatibility (for good however). User
code may need to add extra includes, which were previously resolved indirectly
by including a ROOT header. For example:

  * TBuffer.h - TObject.h doesn't include TBuffer.h anymore. Third party code,
    replying on the definition of TBufer will need to include TBuffer.h, along
    with TObject.h.
  * TSystem.h - for some uses of gSystem.
  * GeneticMinimizer.h
  * ...

Other improvements, which may cause compilation errors in third party code:

  * If you get `std::type_info` from Rtypeinfo.h, `type_info` should be spelled
    `std::type_info`.

Also:

  * `TPluginManager` was made thread-safe [ROOT-7927].
  * On MacOSX, backtraces are now generated without external tools [ROOT-6667].
  * The set of include paths considered by the interpreter has been reduced to the bare minimum.

### Containers

* A pseudo-container (generator) was created, `ROOT::TSeq<T>`. This template is inspired by the xrange built-in function of Python. See the example [here](https://root.cern.ch/doc/master/cnt001__basictseq_8C.html).

### Meta Library

Add a new mode for `TClass::SetCanSplit` (2) which indicates that this class and any derived class should not be split.  This included a rework the mechanism checking the base classes.  Instead of using `InheritsFrom`, which lead in some cases, including the case where the class derived from an STL collection, to spurrious autoparsing (to look at the base class of the collection!), we use a custom walk through the tree of base classes that checks their value of `fCanSplit`.  This also has the side-effect of allowing the extension of the concept 'base class that prevent its derived class from being split' to any user class.  This fixes [ROOT-7972].

### Dictionaries

* Add the -excludePath option to rootcling to forbid dictionaries to remember include paths expressed in the command line invocation.
* Genreflex and rootcling cannot generate capability files anymore.
* Fix [ROOT-7760]: Fully allow the usage of the dylib extension on OSx.
* Fix [ROOT-7879]: Prevent LinkDef files to be listed in a rootmap file and use (as the user actually expects) the header files #included in the linkdef file, if any, as the top level headers.
* Add the *noIncludePaths* switch both for rootcling and genreflex to allow to loose track of the include paths in input to the dictionary generator.
* Fix handling of template parameter pack in the forward declaration printer. [ROOT-8096]
* Do not autoparse headers for classes in the pch.
* Avoid autoparse on IsForeign() if possible.
* Check for new-style empty pcm with key named "EMPTY" created since commit 90047b0cba6fd295f5c5722749a0d043fbc11ea5.
* Do not insert macro definition of `__ROOTCLING__` into the pch.

### Interpreter Library

* llvm / clang have been updated to r274612.
* The GCC5 ABI is now supported [ROOT-7947].
* Exceptions are now caught in the interactive ROOT session, instead of terminating ROOT.
* A ValuePrinter for tuple and pair has been added to visualise the content of these entities at the prompt.
* When interpreting dereferences of invalid pointers, cling will now complain (throw, actually) instead of crash.
* Resolve memory hoarding in some case of looking up functions [ROOT-8145]

## Parallelism

* Three methods have been added to manage implicit multi-threading in ROOT: `ROOT::EnableImplicitMT(numthreads)`, `ROOT::DisableImplicitMT` and `ROOT::IsImplicitMTEnabled`. They can be used to enable, disable and check the status of the global implicit multi-threading in ROOT, respectively.
* Even if the default reduce function specified in the invocation of the `MapReduce` method of `TProcessExecutor` returns a pointer to a `TObject`, the return value of `MapReduce` is properly casted to the type returned by the map function.
* Add a new class named `TThreadExecutor` implementing a MapReduce framework sharing `TProcessExecutor` interface and based in tbb.
* Add a new class named `TExecutor` defining the MapReduce interface for `TProcessExecutor` and `TThreadExecutor`, who inherit from it.
* Remove all `TPool` signatures accepting collections as an argument with the exception of std::vector and initializer_lists.  
* Extend `TThreadExecutor` functionality offering parallel reduction given a binary operator as a reduction function.
* Add a new class named `TThreadedObject` which helps making objects thread private and merging them.
* Add tutorials showing how to fill randomly histograms using the `TProcessExecutor` and `TThreadExecutor` classes.
* Add tutorial showing how to fill randomly histograms from multiple threads.
* Add the `ROOT::TSpinMutex` class, a spin mutex compliant with C++11 requirements.
* Add a new Implicit Multi-Threading (IMT) use case, incarnated in method `TTreeProcessor::Process`. `TTProcessor::Process` allows to process the entries of a TTree in parallel. The user provides a function that receives one parameter, a TTreeReader, that can be used to iterate over a subrange of entries. Each subrange corresponds to a cluster in the TTree and is processed by a task, which can potentially be run in parallel with other tasks.
* Add a new implementation of a RW lock, `ROOT::TRWSpinLock`, which is based on a `ROOT::TSpinMutex`. `TRWSpinLock` tries to make faster the scenario when readers come and go but there is no writer, while still preventing starvation of writers.

## I/O Libraries

* Support I/O of `std::unique_ptr`s and STL collections thereof.
* Support I/O of `std::array`.
* Support I/O of `std::tuple`. The dictionary for those is never auto generated and thus requires explicit request of the dictionary for each std::tuple class template instantiation used, like most other class templates.
* Custom streamers need to #include TBuffer.h explicitly (see [section Core Libraries](#core-libs))
* Check and flag short reads as errors in the xroot plugins. This fixes [ROOT-3341].
* Added support for AWS temporary security credentials to TS3WebFile by allowing the security token to be given.
* Resolve an issue when space is freed in a large `ROOT` file and a TDirectory is updated and stored the lower (less than 2GB) freed portion of the file [ROOT-8055].

- ##### TBufferJSON:
    + support data members with `//[fN]` comment
    + preliminary support of STL containers
    + JSON data can be produced with `TObject::SaveAs()` method


## TTree Libraries

* TChains can now be histogrammed without any C++ code, using the command line tool `rootdrawtree`. It is based on the new class `TSimpleAnalysis`.
* Do not automatically setup read cache during `TTree::Fill()`. This fixes [ROOT-8031].
* Make sure the option "PARA" in `TTree::Draw` is used with at least tow variables [ROOT-8196].
* The with `goff` option one can use as many variables as needed. There no more
  limitation, like with the options `para`and `candle`.
* Fix detection of errors that appears in nested TTreeFormula [ROOT-8218]
* Better basket size optimization by taking into account meta data and rounding up to next 512 bytes, ensuring a complete cluster fits into a single basket.

### Fast Cloning

We added a cache specifically for the fast option of the TTreeCloner to significantly reduce the run-time when fast-cloning remote files to address [ROOT-5078].  It can be controlled from the `TTreeCloner`, `TTree::CopyEntries` or `hadd` interfaces.  The new cache is enabled by default, to update the size of the cache or disable it from `TTreeCloner` use: `TTreeCloner::SetCacheSize`.  To do the same from `TTree::CopyEntries` add to the option string "cachesize=SIZE".  To update the size of the cache or disable it from `hadd`, use the command line option `-cachesize SIZE`.  `SIZE` shouyld be given in number bytes and can be expressed in 'human readable form' (number followed by size unit like MB, MiB, GB or GiB, etc. or SIZE  can be set zero to disable the cache.

### Other Changes

* Update `TChain::LoadTree` so that the user call back routine is actually called for each input file even those containing `TTree` objects with no entries.
* Repair setting the branch address of a leaflist style branch taking directly the address of the struct.  (Note that leaflist is nonetheless still deprecated and declaring the struct to the interpreter and passing the object directly to create the branch is much better).
* Provide an implicitly parallel implementation of `TTree::GetEntry`. The approach is based on creating a task per top-level branch in order to do the reading, unzipping and deserialisation in parallel. In addition, a getter and a setter methods are provided to check the status and enable/disable implicit multi-threading for that tree (see Parallelisation section for more information about implicit multi-threading).
* Properly support `std::cin` (and other stream that can not be rewound) in `TTree::ReadStream`. This fixes [ROOT-7588].
* Prevent `TTreeCloner::CopyStreamerInfos()` from causing an autoparse on an abstract base class.

## Histogram Libraries

* TH2Poly has a functional Merge method.
* Implemented the `TGraphAsymmErrors` constructor directly from an ASCII file.

## Math Libraries


* New template class `TRandomGen<Engine>` which derives from `TRandom` and integrate new random generator engines as TRandom classes.
* New TRandom specific types have been defined for these following generators:
    *  `TRandomMixMax`  - recommended MIXMAX generator with N=240
    *  `TRandomMixMax17` - MIXMAX generator with smaller state (N=17) and faster seeding time
    *  `TRandomMixMax256` - old version of MIXMAX generator (N=256)
    *  `TRandomMT64`  - 64 bit Mersenenne Twister generator from the standard library (based on `std::mt19937_64`). This generates 64 bit random numbers, while `TRandom3` generates only 32 bit random numbers.
    *  `TRandomRanlux48` - 48 bit Ranlux generator. Note that `TRandom1` is a 24 bit generator. 
 * Improve thread safety of `TMinuit` constructor [ROOT-8217]
* Vc has ben removed from the ROOT sources. If the option 'vc' is enabled, the package will be searched (by default),
  alternatively the source tarfile can be downloded and build with the option 'builtin_vc'.

## TMVA Libraries

* New `DataLoader` class that allows flexibility in variable and dataset selection. 
* New Deep Neural Network. Three different versions are available, which can be selected with the 'Architecture' option. See also the tutorial`tmva/TMVAClassification.C` for using the new DNN.
    * `Architecture=STANDARD`  to select the earlier version.
    * `Architecture=CPU` to select the newer version for CPU, but designed also for GPU and optimized for speed and with multi-class support. 
    * `Architecture=GPU` to select the newer GPU version. Requires configuration of ROOT with CUDA or OpenCL enabled. 
* Support for Cross Validation (see tutorial `tmva/TMVACrossValidation` as an example).
* Support for Hyper-Parameter tuning for BDT and SVM methods.
* New Variable Importance algorithm independent of the MVA method.
* New Loss Function class for regression.
* Improvements in the SVM method: new kernel functions.
* New `ROCCurve` class. 
* New interface to Keras (PyKeras)  available in the PyMVA library.
* Support for Jupyter notebooks
    * Support for all the functionality available in GUI: preprocessing, variable correlations, classifier output.
    * New classifier visualization for BDT, ANN and DNN.
    * Interactive training for all methods.


## 2D Graphics Libraries

* In `TColor::SetPalette`, make sure the high quality palettes are defined
  only once taking care of transparency. Also `CreateGradientColorTable` has been
  simplified.
* New fast constructor for `TColor` avoiding to call `gROOT->GetColor()`. The
  normal constructor generated a big slow down when creating a Palette with
  `CreateGradientColorTable`.
* In `CreateGradientColorTable` we do not need anymore to compute the highest
  color index.
* In `TGraphPainter`, when graphs are painted with lines, they are split into
  chunks of length `fgMaxPointsPerLine`. This allows to paint line with an "infinite"
  number of points. In some case this "chunks painting" technic may create artefacts
  at the chunk's boundaries. For instance when zooming deeply in a PDF file. To avoid
  this effect it might be necessary to increase the chunks' size using the new function:
  `TGraphPainter::SetMaxPointsPerLine(20000)`.
* When using line styles different from 1 (continuous line), the behavior of TArrow
  was suboptimal. The problem was that the line style is also applied to the arrow
  head, which is usually not what one wants.
  The arrow tip is now drawn using a continuous line.
* It is now possible to select an histogram on a canvas by clicking on the vertical
  lines of the bins boundaries.
  This problem was reported [here](https://sft.its.cern.ch/jira/browse/ROOT-6649).
* When using time format in axis, `TGaxis::PaintAxis()` may in some cases call
  `strftime()` with invalid parameter causing a crash.
  This problem was reported [here](https://sft.its.cern.ch/jira/browse/ROOT-7689).
* Having "X11.UseXft: yes" activated in .rootrc file and running
  [this](https://sft.its.cern.ch/jira/browse/ROOT-7985) little program,
  resulted in a crash.
* Ease the setting of the appearance of joining lines for PostScript and PDF
  output. `TPostScript::SetLineJoin`
  allowed to set the line joining style for PostScript files. But the setting this
  parameter implied to create a `TPostScript` object. Now a `TStyle` setting has been
  implemented and it is enough to do:
``` {.cpp}
  gStyle->SetLineJoinPS(2);
```
  Also this setting is now active for PDF output.
  This enhancement was triggered by [this forum question](https://root.cern.ch/phpBB3/viewtopic.php?f=3&t=21077).
* Make sure the palette axis title is correct after a histogram cloning. This
  problem was mentioned [here](https://sft.its.cern.ch/jira/browse/ROOT-8007).
* `TASImage` When the first or last point of a wide line is exactly on the
  window limit the line is drawn vertically or horizontally.
  This problem was mentioned [here](https://sft.its.cern.ch/jira/browse/ROOT-8021)
* Make sure that `TLatex` text strings containing "\\" (ie: rendered using `TMathText`)
  produce an output in PDF et SVG files.
* In TLatex, with the Cocoa backend on Mac the Angstroem characters did not render correctly.
  This problem was mentioned [here](https://root.cern.ch/phpBB3/viewtopic.php?f=3&t=21321)
* New version of libpng (1.2.55) as requested [here](https://sft.its.cern.ch/jira/browse/ROOT-8045).
* Enhancement of the CANDLE drawing option (implemented by Georg Troska georg.troska@tu-dortmund.de).
  This option has been completely rewritten and offers a wide range of possibilities.
  See the THistPainter reference guide for all the details and examples.
* Fix `TText` copy constructor as requested [here](https://sft.its.cern.ch/jira/browse/ROOT-8116).
  New example to check this fix.
* SVG boxes were not correct when `x2<1` (reported [here](https://sft.its.cern.ch/jira/browse/ROOT-8126)).
* In TASImage there was no protection against graphics being drawn outside the assigned
  memory. That may generate some crashes like described [here](https://sft.its.cern.ch/jira/browse/ROOT-8123).
* In TASImage: transparent rectangles did not work when png files were created in batch mode.
* In TASImage: implement transparent text for png files created in batch mode.
* TCanvas title was not set correctly when a TCanvas was read from a TFile.
  (reported [here](https://root.cern.ch/phpBB3/viewtopic.php?f=3&t=21540&p=94053#p93981)).
* The text generated by `TSVG` has now the `xml:space="preserve"` attribute in order
  to be editable later on using external softwares like "inkscape". This improvement
  was suggested [here](https://sft.its.cern.ch/jira/browse/ROOT-8161).
* In TLatex with the Cocoa backend on Mac the `#tilde` position was too low.
* New optional parameter "option" in `TPad::BuildLegend` to set the TLegend option (Georg Troska).
* TCandle: a new candle plot painter class. It is now used in THistPainter and THStack
  to paint candle plots (Georg Troska).
* Fix two issues with the fill patterns in `TTeXDump` (reported [here](https://sft.its.cern.ch/jira/browse/ROOT-8206)):
    - The pattern number 3 was not implemented.
    - Filled area drawn with pattern where surrounded by a solid line.
* Support custom line styles in `TTeXDump` as requested [here](https://sft.its.cern.ch/jira/browse/ROOT-8215)
* `TColor::GetFreeColorIndex()` allows to make sure the new color is created with an
  unused color index.
* In `TLegend::SetHeader` the new option `C` allows to center the title.
* New method `ChangeLabel` in `TGaxis` and `TAxis`allowing to a fine tuning of
  individual labels attributes. All the attributes can be changed and even the
  label text itself. Example:
``` {.cpp}
  {
     c1 = new TCanvas("c1","Examples of Gaxis",10,10,900,500);
     c1->Range(-6,-0.1,6,0.1);
     TGaxis *axis1 = new TGaxis(-5.5,0.,5.5,0.,0.0,100,510,"");
     axis1->SetName("axis1");
     axis1->SetTitle("Axis Title");
     axis1->SetTitleSize(0.05);
     axis1->SetTitleColor(kBlue);
     axis1->SetTitleFont(42);
     axis1->ChangeLabel(1,-1,-1,-1,2);
     axis1->ChangeLabel(3,-1,0.);
     axis1->ChangeLabel(5,30.,-1,0);
     axis1->ChangeLabel(6,-1,-1,-1,3,-1,"6th label");
     axis1->Draw();
  }
```
  Being available in `TAxis`, this method allow to change a label on and histogram
  plot like:
``` {.cpp}
   hpx->Draw();
   hpx->GetXaxis()->ChangeLabel(5,-1,-1,-1,kRed,-1,"Zero");
```
* New class `TAxisModLab`: a  TAxis helper class used to store the modified labels.
* `TPie` the format parameter set by `SetPercentFormat` was ignored.
  (reported [here](https://sft.its.cern.ch/jira/browse/ROOT-8294))
* Improvements in the histogram plotting option `TEXT`: In case several histograms
  are drawn on top ot each other (using option `SAME`), the text can be shifted
  using `SetBarOffset()`. It specifies an offset for the  text position in each
  cell, in percentage of the bin width.
* `TGaxis::PaintAxis()` might caused a correctness problem in multithreaded
   context when handling optionTime with `%F`. This was reported
   [here](https://sft.its.cern.ch/jira/browse/ROOT-8309). The fixed was suggested
   by Philippe Gras (philippe.gras@cea.fr).
* `TGaxis::PaintAxis()` misplaced the `x10` at the end of the axis for non vertical
   or horizontal axis
   [here](https://root.cern.ch/phpBB3/viewtopic.php?f=3&t=22363).

## 3D Graphics Libraries

* When painting a `TH3` as 3D boxes, `TMarker3DBox` ignored the max and min values
  specified by `SetMaximum()` and `SetMinimum()`.
* In `TMarker3DBox` when a box marker has a size equal to zero it is not painted.
  Painting it produced a dot with the X11 backend.
* New class `TRatioPlot` implemented by Paul Gessinger <hello@paulgessinger.com>.
  Class for displaying ratios, differences and fit residuals.

  `TRatioPlot` has two constructors, one which accepts two histograms, and is responsible
  for setting up the calculation of ratios and differences. This calculation is in part
  delegated to `TEfficiency`. A single option can be given as a parameter, that is
  used to determine which procedure is chosen. The remaining option string is then
  passed through to the calculation, if applicable.

  Several examples illustrate how to use this class. See:
  `$ROOTSYS/tutorials/hist/ratioplot?.C`

* New option "I" allowing to draw TGraph with invisible axis (used by `TRatioPlot`);

## New histogram drawing options

### COL2
  COL2 is a new rendering technique providing potential performance improvements
  compared to the standard COL option. The performance comparison of the COL2 to
  the COL option depends on the histogram and the size of the rendering region in
  the current pad. In general, a small (approx. less than 100 bins per axis),
  sparsely populated TH2 will render faster with the COL option.

  However, for larger histograms (approx. more than 100 bins per axis) that are
  not sparse, the COL2 option will provide up to 20 times performance improvements.
  For example, a 1000x1000 bin TH2 that is not sparse will render an order of
  magnitude faster with the COL2 option.

  The COL2 option will also scale its performance based on the size of the pixmap
  the histogram image is being rendered into. It also is much better optimized for
  sessions where the user is forwarding X11 windows through an `ssh` connection.

  For the most part, the COL2 and COLZ2 options are a drop in replacement to the COL
  and COLZ options. There is one major difference and that concerns the treatment of
  bins with zero content. The COL2 and COLZ2 options color these bins the color of zero.

  This has been implemented by Jeromy Tompkins <Tompkins@nscl.msu.edu>

## Geometry Libraries
  A new module geom/vecgeom was introduced to give transparent access to VecGeom 
  solid primitives. VecGeom is a high performance geometry package (link) providing 
  SIMD vectorization for the CPU-intensive geometry algorithms used for geometry
  navigation. The module creates a new library libConverterVG.so depending on the
  VecGeom main library and loaded using the ROOT plug-in mechanism.

  The main functionality provided by the new vecgeom module is to make a conversion 
  in memory of all the shapes in a loaded TGeo geometry into a special adapter
  shape TGeoVGShape, redirecting all navigation calls to the corresponding VecGeom 
  solid. The library loading and geometry conversion can be done with a single call 
  `TVirtualGeoConverter::Instance()->ConvertGeometry()`
  

  After the conversion is done, all existing TGeo functionality is available as for
  a native geometry, only that most of the converted solids provide better navigation 
  performance, despite the overhead introduced by the new adapter shape.

  Prerequisites: installation of VecGeom. 
  The installation instructions are available at <http://geant.web.cern.ch/content/installation>
  Due to the fact that VecGeom provides for the moment static libraries 
  and depends on ROOT, is is advised to compile first ROOT without VecGeom support, 
  then compile VecGeom against this ROOT version, then re-configure ROOT to enable 
  VecGeom and Vc support, using the flags -Dvc=ON -Dvecgeom=on
  
  This has been implemented by Mihaela Gheata <Mihaela.Gheata@cern.ch>

## Database Libraries

* Fix `TPgSQLStatement::SetBinary` to actually handle binary data (previous limited to ascii).

## Networking Libraries

* When seeing too many requested ranges, Apache 2.4 now simply sends the whole file
  (MaxRanges configuration parameter). TWebFile can handle this case now, but this can
  trigger multiple transmissions of the full file. TWebFile warns when Apache reacts by
   sending the full file.


## GUI Libraries

* A new `Browser.ExpandDirectories` option (the default is `yes`) has been added, allowing to prevent expanding the parent directory tree in the ROOT Browser (for example on nfs).


## Language Bindings

### PyROOT

  * Added a new configuration option to disable processing of the rootlogon[.py|C] macro in addition
    ro the -n option in the command arguments. To disable processing the rootlogon do the following
    before any other command that will trigger initialization:
    ```
    >>> import ROOT
    >>> ROOT.PyConfig.DisableRootLogon = True
    >>> ...
    ```

### Notebook integration

  * Refactoring of the Jupyter integration layer into the new package JupyROOT.
  * Added ROOT [Jupyter Kernel for ROOT](https://root.cern.ch/root-has-its-jupyter-kernel)
    * Magics are now invoked with standard syntax "%%", for example "%%cpp".
    * The methods "toCpp" and "toPython" have been removed.
  * Factorise output capturing and execution in an accelerator library and use ctypes to invoke functions.
  * When the ROOT kernel is used, the output is consumed progressively
  * Capture unlimited output also when using an IPython Kernel (fixes [ROOT-7960])

## JavaScript ROOT

- New geometry (TGeo) classes support:
    - browsing  through volumes hieararchy
    - changing visibility flags
    - drawing of selected volumes
    - support of large (~10M volumes) models, only most significant volumes are shown
    - one could activate several clip planes (only with WebGL)
    - interaction with object browser to change visibility flags or focus on selected volume
    - support of floating browser for TGeo objects
    - intensive use of HTML Worker to offload computation tasks and keep interactivity
   - enable more details when changing camera position/zoom
- Improvements in histograms 3D drawing
    - all lego options: lego1..lego4, combined with 'fb', 'bb', '0' or 'z'
    - support axis labels on lego plots
    - support lego plots for TH1
- Significant (up to factor 10) performance improvement in 3D-graphics
- Implement ROOT6-like color palettes
- Support non-equidistant bins for TH1/TH2 objects.
- Improve TF1 drawing - support exp function in TFormula, fix errors with logx scale, enable zoom-in, (re)calculate function points when zooming
- Introduce many context menus for improving interactivity
- Implement col0 and col0z draw option for TH2 histograms, similar to ROOT6
- Implement box and hbox draw options for TH1 class
- Significant (factor 4) I/O performance improvement
- New 'flex' layout:
    - create frames like in Multi Document Interface
    - one could move/resize/minimize/maximize such frames

For more details, like the complete change log, the documentation, and very detailed examples, see the [JSROOT home page](https://root.cern.ch/js) and the [JSROOT project github page](https://github.com/linev/jsroot) 


## Tutorials
* New tutorial `treegetval.C` illustrating how to retrieve  `TTree` variables in arrays.
* Add script to automatically translate tutorials into notebooks
    * Embed it into the documentation generation
    * Make the notebooks available in the [tutorials section of the class documentation](https://root.cern/doc/master/group__Tutorials.html)

## Build, Configuration and Testing Infrastructure
- `root-config` does not suppress deprecation warnings (-Wno-deprecated-declarations) anymore. This means compilers will now diagnose the use of deprecated interfaces in user code.
- Added new 'builtin_vc' option to bundle a version of Vc within ROOT.
  The default is OFF, however if the Vc package is not found in the system the option is switched to
  ON if the option 'vc' option is ON.
- Many improvements (provided by Mattias Ellert):
    - Build RFIO using dpm libraries if castor libraries are not available
    - Add missing glib header path in GFAL module for version > 2
    - Search also for globus libraries wouthout the flavour in the name
    - Add missing io/hdfs/CMakeLists.txt
    - net/globusauth has no installed headers - remove ROOT_INSTALL_HEADERS()
    - Add missing pieces to the cmake config that are built by configure: bin/pq2, bin/rootd, bin/xpdtest, initd and xinitd start-up scripts
    - Only link to libgfortranbegin.a when it is provided by the compiler
    - Don't remove -Wall without also removing -Werror=*
    - Don't overwrite the initial value of CMAKE_Fortran_FLAGS. Inconsistent case variant of CMAKE_Fortran_FLAGS
    - Use the same sonames in cmake as in configure
    - Allow building for ppc64 as well as ppc64le
    - Add build instructions for 32 bit ARM
    - Add build instructions for System Z (s390 and s390x)
    - Make sure that the roots wrapper can be executed
    - Move gl2ps.h to its own subdir
- Added new 'builtin-unuran' option (provided by Mattias Ellert)
- Added new 'builtin-gl2ps' option (provided by Mattias Ellert)
- Added new 'macos_native' option (only for MacOS) to disable looking for binaries, libraires and headers for dependent
  packages at locations other than native MacOS installations. Needed when wanting to ignore packages from Fink, Brew or Ports.
- Added new 'cuda' option to enable looking for CUDA in the system.


