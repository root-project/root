% ROOT Version 6.40 Release Notes
% 2026-5
<a name="TopOfPage"></a>

## Introduction

For more information, see:

[http://root.cern](http://root.cern)

The following people have contributed to this new version:

 Sebastian Alba Vives, Instituto Tecnologico de Costa Rica (TEC),\
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
 David Lange, CERN and Princeton,\
 Sergey Linev, GSI Darmstadt,\
 Lorenzo Moneta, CERN/EP-SFT,\
 Christian Ng, https://laserbear.org,\
 Vincenzo Eduardo Padulano, CERN/EP-SFT,\
 Giacomo Parolini, CERN/EP-SFT,\
 Danilo Piparo, CERN/EP-SFT,\
 Jonas Rembser, CERN/EP-SFT,\
 Silia Taider, CERN/EP-SFT,\
 Devajith Valaparambil Sreeramaswamy, CERN/EP-SFT,\
 Vassil Vassilev, Princeton,\
 Sandro Wenzel, CERN/EP-ALICE,\
 Ned Ganchovski, Proektsoft EOOD,\

## Deprecations

* The headers in `RooStats/HistFactory` for data classes related to the measurement definition were merged into the `RooStats/HistFactory/Measurement.h` header to simplify usage and development. For now, the whole set of header files is kept for backwards compatibility, but the empty headers will be removed in ROOT 7.
* The `TROOT::GetSourceDir()` method is deprecated and will be removed in ROOT 6.42. It stopped making sense because the ROOT source is generally not shipped alongside ROOT in the `src/` subdirectory anymore.
* Using the `rpath` build option - deprecated and without effect since ROOT 6.38 - is now scheduled to give configuration errors starting from ROOT 6.42.
* `TDirectory::AddDirectoryStatus()` and `TDirectory::AddDirectory()` have been deprecated. These functions were meant to replace TH1::AddDirectoryStatus(), but
  never had any effect on ROOT. The associated bit TDirectory::fgAddDirectory was deprecated as well. Although users can set and read the bit, its usage should be
  stopped completely to avoid any confusion. The bit and functions will be removed in ROOT 7.
* The method `RooRealVar::removeRange()` and the corresponding method in `RooErrorVar` have been deprecated because the name was misleading, and they will be removed in ROOT 6.42. Despite the name, the function did not actually remove a range, but only cleared its limits by setting them to  `−inf,+inf` leaving the named range itself defined (so `hasRange()` would still return `true`). Users should now explicitly call `removeMin()` and `removeMax()` to remove the lower and upper limits of a range.
* The `builtin_zeromq` and `builtin_cppzmq` build options are deprecated and will be removed in ROOT 6.42.
  The ZeroMQ library and its C++ bindings are used by the experimental RooFit multiprocessing package, enabled by the `roofit_multiprocess` build option.
  The ZeroMQ versions it requires (>=4.3.6 or 4.3.5 with the draft API) are now available in the package managers of several platforms, for example Conda, Homebrew, Fedora and the Extra Packages for Enterprise Linux (EPEL).
  The `roofit_multiprocess` feature is only required by a small set of RooFit power uses, who are using one of these environments and therefore don't require the builtin ZeroMQ library.
* The overloads of `RooAbsReal::createChi2()` and `RooAbsReal::chi2FitTo()` that take unbinned **RooDataSet** data objects are deprecated and will be removed in ROOT 6.42.
  These methods implemented a specialized chi-square fit for x-y-data with errors in y and optional errors in x, which is conceptually different from the standard histogram-based chi-square in the **RooDataHist** case and can lead to ambiguous results.
  To fit 2D data with errors in and `x` and `y`, use specialized tools like `TGraphErrors::Fit()`, or build an explicit likelihood model if you want to stay with RooFit.
* The **RooStats::HybridPlot** class and the related **HybridResult::GetPlot** method are deprecated and will be removed in ROOT 6.42.
  We kindly ask users to write their own ROOT-based plotting code, possibly based on the source code of the deprecated **HybridPlot** class, as pre-existing plot helpers are usually failing to be flexible enough for large-scale adoption.
* The `TGrid*` family of abstract classes provided the basis for accessing GRID services from ROOT. All the concrete plugins (AliEn, glite, etc.) were removed years ago. These facilities should now be unused. The classes will be removed in ROOT 6.42.
* The `TFTP`, `TNetFile`, `TNetFileStager`, and `TNetSystem` classes are deprecated and will be removed in ROOT 6.42. These classes rely on **rootd**, which was removed in release 6.16.
* The ROOT **auth** package together with `TVirtualAuth` and `TROOT::GetListOfSecContexts()`, and the **authenticated sockets** (`TSocket::CreateAuthSocket()`) feature are deprecated and will be remove in ROOT 6.42.
  The security assumtions in the current socket authentication implementation is not up to date anymore.
  Secure communication should be provided by standard means, such as SSL sockets or SSH tunneling.
* The `builtin_davix` build option has been removed.
  The Davix I/O code in ROOT remains uneffected and is built as before provided that the Davix library is found on the system.
* `RRealField::SetQuantized` now has a new overload and the existing signature has been deprecated. The new overload enforces proper ordering of the arguments.
  The deprecated overload will be removed in ROOT 6.42.
* The bindings to the R programming language that are enabled with the `r=ON` build option (`TRInterface` and friends) are deprecated and will be removed in ROOT 6.42. Their maintenance is no longer justified, given the broader adoption of the scientific Python ecosystem. Users who still rely on R from C++ are encouraged to call R directly via https://cran.r-project.org/package=RInside, which is what the ROOT bindings were using internally.

## Removals

* The `TH1K` class was removed. `TMath::KNNDensity` can be used in its stead.
* The `TObject` equality operator Pythonization (`TObject.__eq__`) that was deprecated in ROOT 6.38 and scheduled for removal in ROOT 6.40 is removed.
* Comparing C++ `nullptr` objects with `None` in Python now raises a `TypeError`, as announced in the ROOT 6.38 release notes. Use truth-value checks like `if not x` or `x is None` instead.
* The `TGLIncludes.h` and `TGLWSIncludes.h` that were deprecated in ROOT 6.38 and scheduled for removal are gone now. Please include your required headers like `<GL/gl.h>` or `<GL/glu.h>` directly.
* The GLEW headers (`GL/eglew.h`, `GL/glew.h`, `GL/glxew.h`, and `GL/wglew.h`) that were installed when building ROOT with `builtin_glew=ON` are no longer installed. This is done because ROOT is moving away from GLEW for loading OpenGL extensions.
* The `TF1`, `TF2`, and `TF3` constructors for CINT compatibility were removed. This concerns the templated constructors that additionally took the name of the used functor class and member function. With ROOT 6, these names can be omitted.
* The `TMultiGraph::Add(TMultiGraph*, Option_t*)` overload that adds the graphs in another **TMultiGraph** to a **TMultiGraph** is removed without deprecation.
  It was inconsistent from a memory ownership standpoint.
  A **TMultiGraph** always owns all the added graphs, so adding the same graph instances to two **TMultiGraphs** forcibly led to double-deletes.
  If you want to add all graphs from `otherMultiGraph` to `multiGraph`, please use a for-loop and clone the graphs instead:
  ```c++
  for (TObject *gr : *otherMultiGraph) {
     multiGraph->Add(static_cast<TGraph*>(gr->Clone()));
  }
  ```
* The `compression_default` build option was removed. It was supposed to change the default compression algorithm, but didn't actually work with the default parameters of `TFile`.

## Build System

### Moving from builtin dependencies to system-provided packages

* The general direction of the ROOT project is to become more and more reliant on system packages. It is *recommended* to make the packages required by ROOT available on the system, e.g. via a package manager, and not with the builtin mechanism. This allows for timely updates and reduces the size of the installed binaries.
* The previously vendored builtins `freetype`, `zlib`, `lzma`, `zstd`, `lz4`, `libpng`, `giflib`, `libjpeg`, and `openssl` should be installed in the system if possible. ROOT will not automatically fall-back to their builtin versions if these are not found: the user is informed of that with a helpful message. If installing these dependencies in the system is not possible, the CMake option `-Dbuiltin_XYZ=ON` has to be consciously chosen by the user.
* For the builtin versions of `freetype`, `zlib`, `lzma`, `zstd`, `lz4`, `libpng`, `giflib`, `libjpeg`, the source tarballs are now fetched from [SPI](https://spi.web.cern.ch)'s [website](https://lcgpackages.web.cern.ch/), as for the vast majority of ROOT's builtins, e.g. `openssl` or `xrootd`.

## Core Libraries

* ROOT now adds a RUNPATH to compiled macros. This ensures that when compiled macros are loaded, they load the libraries that belong to the ROOT installation
  that compiled the macro. See [TSystem::SetMakeSharedLib()](https://root.cern.ch/doc/master/classTSystem.html#a80cd12e064e2285b35e9f39b5111d20e) for
  customising or disabling the RUNPATH.
* `rootcling` fails if no selection rule is specified and if the creation of a C++ module is not requested.
* To ease debugging of unwanted auto-parsing triggered by TClass::GetClass, two new features are introduced:
* * Give access to the list of classes that triggered auto-parsing:
```
// Print the list
gInterpreter->Print("autoparsed");
// Get the list/set:
((TCling*)gInterpreter)->GetAutoParseClasses();
```
* * Auto-parsing of header files can now be explicitly disabled during the execution of TClass::GetClass;
for example, this can be used to enforce that no header is loaded for I/O operations. To disable the
auto-parsing during `TClass::GetClass`, you can either set the shell environment variable
`ROOT_DISABLE_TCLASS_GET_CLASS_AUTOPARSING` (to anything) or set the `rootrc` key `Root.TClass.GetClass.AutoParsing` to `false`.
* Alignment of classes and numerical types is now recorded in the dictionary and propagated through `TClass`, `TDataType`, and `TStreamerElement`.  `TStreamerInfo::BuildOld` uses this information to correctly lay out emulated classes and older-version StreamerInfos (for the input of I/O customization), ensuring proper alignment for over-aligned types.
  Fixes [#21667](https://github.com/root-project/root/issues/21667).
  (PR [#21669](https://github.com/root-project/root/pull/21669))

## Geometry

* The list of logical volumes gets now rehashed automatically, giving an important performance improvement for setups having a large number of those.
* `TGeoTessellated` now has efficient, BVH accelerated, navigation function implementations. This makes it possible to use `TGeoTessellated` in applications using `TGeoNavigator` (such as detector simulation).

### Extensible color schemes for geometry visualization
ROOT now provides an extensible mechanism to assign colors and transparency to geometry volumes via the new `TGeoColorScheme` strategy class, used by `TGeoManager::DefaultColors()`.

This improves the readability of geometries imported from formats such as GDML that do not store volume colors. The default behavior now uses a name-based material classification (e.g. metals, polymers, composites, gases) with a Z-binned fallback. Three predefined color sets are provided:
* `EGeoColorSet::kNatural` (default): material-inspired colors
* `EGeoColorSet::kFlashy`: high-contrast, presentation-friendly colors
* `EGeoColorSet::kHighContrast`: darker, saturated colors suited for light backgrounds

Users can customize the behavior at runtime by providing hooks (std::function) to override the computed color, transparency, and/or the Z-based fallback mapping.

**Usage examples:**
```cpp
gGeoManager->DefaultColors(); // default (natural) scheme

TGeoColorScheme cs(EGeoColorSet::kFlashy);
gGeoManager->DefaultColors(&cs); // select a predefined scheme
```

**Override examples (hooks):**
```cpp
TGeoColorScheme cs(EGeoColorSet::kNatural);
cs.SetZFallbackHook([](Int_t Z, EGeoColorSet) -> Int_t {
   float g = std::min(1.f, Z / 100.f);
   return TColor::GetColor(g, g, g); // grayscale fallback
});
gGeoManager->DefaultColors(&cs);
```

A new tutorial macro demonstrates the feature and customization options: `tutorials/visualization/geom/geomColors.C`.

See: https://github.com/root-project/root/pull/21047 for more details

### Accelerated overlap checking with parallel execution
The geometry overlap checker (TGeoChecker::CheckOverlaps) has been significantly refactored and optimized to improve performance and scalability on large detector geometries.

Overlap checking is now structured in three explicit stages:

1. Candidate discovery
Potentially overlapping volume pairs are identified using oriented bounding-box (OBB) tests, drastically reducing the number of candidates to be examined.

2. Surface point generation and caching
Points are generated on the surfaces of the candidate shapes (including additional points on edges and generators) and cached per shape.
The sampling density can be tuned via:
* `TGeoManager::SetNsegments(nseg)` (default: 20)
* `TGeoManager::SetNmeshPoints(npoints)` (default: 1000)

3. Overlap and extrusion checks
The actual geometric checks are performed using navigation queries.
This stage is now **parallelized** and automatically uses ROOT’s implicit multithreading when enabled.

Only the final stage is currently parallelized, but it dominates the runtime for complex geometries and shows good strong scaling.

For large assembly-rich detector descriptions such as the ALICE O² geometry, the new candidate filtering reduces the number of overlap candidates by roughly three orders of magnitude compared to the legacy implementation. Combined with multithreaded execution, this reduces the total runtime of a full overlap check from hours to minutes on modern multi-core systems.

**Usage example**

```cpp
ROOT::EnableImplicitMT();        // enable parallel checking
gGeoManager->SetNsegments(40);  // increase surface resolution if needed
gGeoManager0->SetNmeshPoints(2000); // increase resolution of points on surface-embedded segments if needed
gGeoManager->CheckOverlaps(1.e-6);
```
Performance and scaling plots for the CMS Run4 and ALICE aligned geometry are included in the corresponding pull request.

See: https://github.com/root-project/root/pull/20963 for implementation details and benchmarks

## I/O

* The behavior or `TDirectoryFile::mkdir` (which is also `TFile::mkdir`) was changed regarding the creation of directory hierarchies: calling `mkdir("a/b/c", "myTitle")` will now assign `myTitle` to the innermost directory `"c"` (before this change it would assign it to `"a"`).
* Fixed a bug in `TDirectoryFile::mkdir` where passing `returnExistingDirectory = true` would not work properly in case of directory hierarchies. The option is now correctly propagated to `mkdir`'s inner invocations.

### File Permissions Now Respect System `umask`

ROOT now respects the system `umask` when creating files, following standard Unix conventions.

**Previous behavior:** Files were created with hardcoded `0644` permissions (owner read/write, group/others read-only), ignoring the system `umask`.

**New behavior:** Files are created with `0666` permissions masked by the system `umask` (`0666 & ~umask`), consistent with standard Unix file creation functions like `open()` and `fopen()`.

**Impact:**
- **Users with default `umask 022`** (most common): No change - files are still created as `0644`
- **Users with stricter `umask` values** (e.g., `0077`): Files will now be created with more restrictive permissions (e.g., `0600` - user-only access)
- **Users with permissive `umask` values** (e.g., `0002`): Files may be created with slightly more open permissions (e.g., `0664` - group-writable)

**Note:** If you require specific file permissions regardless of `umask`, you can set `umask` explicitly before running ROOT (e.g., `umask 022`) or use `chmod` after file creation.

This change affects the following classes:  `TFile`, `TMapFile`, `TMemFile`, `TDCacheFile`, `TFTP`, and `TApplicationServer`.

### TTree

### RNTuple

- A new API to create "active entry tokens" was added to the `RNTupleReader`. Active entry tokens that are used prevent cache eviction of the entries at hand.
This should be used for multi-stream reading where multiple threads share a single reader.
- `RRawPtrWriteEntry` is now part of the stable API, in the `ROOT::Detail` namespace. It is useful for frameworks passing data to RNTuple as `const` raw pointers.
- RNTuple reads into a memory-adopted `RVec` (instead of turning it into an owning `RVec`) if the vector size matches the size of the collection on disk.
- Enable XRootD redirection when opening an RNTuple from a file on EOS
- Python API: Both the reader's and writer's context objects will close the underlying file upon `__exit__`. The object will raise an exception if accessed after the `with` statement.
- Several improvements to avoid header auto-parsing caused by RNTuple reading classes.
- Various performance and memory improvements.
- Initial support for SoA layout support was added as an experimental feature. Dictionaries for classes can be annotated as SoA layout classes of an underlying record type through the `rntupleSoARecord(<underlying record type>)` dictionary attribute. A new `RSoAField` class was added. Currently, only writing is supported. To read, users need to impose a model or use views. Read support for the SoA field will follow.
- Support for RNTuple "attributes" was added as an experimental feature. RNTuple attributes are used to store meta-data associated with a main RNTuple.

### HTTP I/O

- There is a new HTTP I/O implementation in ROOT based on libcurl. It provides modern HTTP remote read capabilities (HTTP/2, compression, secure transport, etc.) on Linux, macOS, and Windows. The new HTTP I/O can also connect to protected resources on S3 enpoints using the `S3_ACCCESS_KEY`, `S3_SECRET_KEY`, and `S3_REGION` environment variables. By default, the new HTTP I/O is automaticlly built if libcurl is found on the system (manual cmake option: `-Dcurl=[on|off]`). If both, Davix based and libcurl based HTTP plugins are available, the Davix plugin has precedence unless `Curl.ReplaceDavix: yes` is set in the `.rootrc`.

## Math

### Migration from VecCore/Vc to `std::experimental::simd` for Vectorization

We have migrated the vectorized backends of **TMath** and **TFormula** from **VecCore/Vc** to `std::experimental::simd`, where available.

On Linux, `std::experimental::simd` is assumed to be available when ROOT is compiled with C++20 or later, which in practice corresponds to sufficiently recent GCC and Clang compilers. To keep the build system simple and robust, ROOT does not explicitly check compiler versions: users opting into C++20 are expected to use modern toolchains.

**Impact on Linux users**

ROOT builds with C++17 on Linux no longer provide vectorized TMath and TFormula.
This is an intentional and accepted trade-off of the migration. These vectorized features were rarely used, while maintaining them significantly complicated the codebase and build configuration.

Users who rely on vectorized **TMath** or the vectorized **TFormula** backend are encouraged to build ROOT with C++20.
Doing so restores vectorization through `std::experimental::simd`, providing a more robust and future-proof implementation.

**Windows and Apple silicon users are unaffected**

VecCore/Vc did not work on Windows previously, and Vc never provided production-ready support for ARM/Neon, so Apple silicon did not benefit from vectorization before this change.

**Build system changes**

As part of this migration, the following build options are deprecated. From ROOT 6.42, setting them will result in configuration errors.

  * `vc`
  * `veccore`
  * `builtin_vc`
  * `builtin_veccore`

### Added Modified Anderson-Bjork (ModAB) root-finding algorithm

This is new and efficient bracketing root-finding [(Ganchovski, et al. 2026)](https://www.mdpi.com/1999-4893/19/5/332). It combines bisection with Anderson-Bjork's method to achieve superlinear convergence (e.i. = 1.7-1.8), while preserving worst-case optimality. According to the benchmarks, it outperforms classical algorithms like ITP, Brent and Ridders. It is implemented as the `ROOT::Math::ModABRootFinder` class.

## RooFit

### General changes

- A new RooAbsPdf has been added: `RooStudentT`, which describes the location-scale student's t-distribution.
- The `RooNumber::setRangeEpsRel()` and `RooNumber::setRangeEpsAbs()` have been
  introduced 2 years ago in 48637270a9113aa to customize range check behavior
  to be like before ROOT 6.28, but this has not been proven necessary. Instead,
  looking up the static variables with the epsilon values incurred significant
  overhead in `RooAbsRealLValue::inRange()`, which is visible in many-parameter
  fits. Therefore, these functions are removed.
- The constructors of **RooDataSet** and **RooDataHist** that combine datasets via `Index()` and `Import()` now validate that the import names correspond to existing states of the index category. If an imported data slice refers to a category label that is not defined in the index category, the constructor now throws an error.
  Previously, such labels were silently added as new category states, which could lead to inconsistent datasets when the state names were not synchronized with the model definition. This change prevents the creation of invalid combined datasets and surfaces configuration problems earlier.

### ONNX model integration via RooONNXFunc

A new class `RooONNXFunc` has been introduced to enable the use of machine learning models in ONNX format directly within RooFit workflows.

`RooONNXFunc` wraps an ONNX model as a `RooAbsReal`, allowing it to be used as a building block in likelihoods, fits, and statistical analyses without additional boilerplate code. The class supports models with one or more statically-shaped input tensors and a single scalar output.
The class was designed to share workspaces with neural functions for combined fits in RooFit-based frameworks written in C++.
Therefore, the `RooONNXFunc` doesn't depend on any Python packages and fully supports ROOT IO,

**Key features:**

  * **Compiled inference via TMVA SOFIE:** The ONNX graph is translated into optimized C++ code at runtime using SOFIE, avoiding external runtime dependencies.

  * **Automatic differentiation with Clad:** Gradients of the model output with respect to RooFit parameters are generated automatically for efficient gradient-based minimization with RooFits `"codegen"` backend.

  * **Portable serialization:** The ONNX model is stored as part of the `RooONNXFunc` object and serialized with ROOT I/O. Upon reading a workspace, the inference code is regenerated automatically.

### Deprecation of the the constant term optimization for legacy test statistic classes

The **RooFit::Optimize()** option (constant term optimization) has been deprecated and will be removed in ROOT 6.42.
This option only affects the `legacy` evaluation backend.

**Important behavior change**: Constant term optimization is now disabled by default when using the legacy backend.
Previously, it was enabled by default. As a result, users who still rely on the legacy backend may observe *slower fits*.

The default vectorized CPU evaluation backend (introduced in ROOT 6.32) already performs these optimizations automatically and is not affected by this change.
Users are strongly encouraged to switch to the vectorized CPU backend if they are still using the legacy backend.

If the vectorized backend does not work for a given use case, **please report it by opening an issue on the ROOT GitHub repository**.

### New implementation of `RooHistError::getPoissonInterval`

**RooHistError::getPoissonInterval** was reimplemented to use an exact chi-square–based construction (Garwood interval) instead of asymptotic approximations and lookup tables.

Previously:

  * For `n > 100`, a hard-coded asymptotic formula was used, which was not explained in the documentation.
  * That approximation corresponds to inverting the Poisson score test and was only correct for `nSigma = 1`.
  * The behavior for `n > 100` was not statistically consistent because of the hard transition between exact formula and approximation, resulting in a discrete and unexpected jump in approximation bias.
  * For small `n`, numerical root finding and a lookup table were used.

Now:

  * The interval is computed directly using chi-square quantiles for all `n`.
  * The construction provides exact frequentist coverage.
  * The implementation works consistently for arbitrary `nSigma`.
  * The hard-coded `n > 100` threshold, lookup table, and numerical root finding were removed.
  * For `n = 0`, a one-sided upper limit is used (with lower bound fixed at 0), consistent with the physical constraint `mu ≥ 0`.

Results may differ from previous ROOT versions for `n > 100` or `nSigma != 1`.
The new implementation is statistically consistent and recommended.

### Removal of global expensive object caching in RooFit

The global “expensive object cache” used in RooFit to store numeric integrals and intermediate histogram results has been removed.

While originally intended as a performance optimization, this mechanism could lead to **incorrect results due to cache collisions**: cached integrals or histograms created in one context (e.g. during plotting) could be reused unintentionally in a different context, even when the underlying configuration had changed.

Given the risk of silently incorrect physics results, and the absence of known workflows that depend critically on this feature, this global caching mechanism has been removed. If you encounter performance regressions in workflows involving expensive integrals or convolutions, we encourage you to report your use case and performance regression as a GitHub issue, so that targeted and robust optimizations can be developed,

## RDataFrame

- The change of default compression settings used by Snapshot for the TTree output data format introduced in 6.38 (was 101 before 6.38, became 505 in 6.38) is reverted. That choice was based on evidence available up to that point that indicated that ZSTD was outperforming ZLIB in all cases for the available datasets. New evidence demonstrated that this is not always the case, and in particular for the notable case of TTree branches made of collections where many (up to all) of them are empty. The investigation is described at https://github.com/vepadulano/ttree-lossless-compression-studies. The new default compression settings for Snapshot are respectively `kUndefined` for the compression algorithm and `0` for the compression level. When Snapshot detects `kUndefined` used in the options, it changes the compression settings to the new defaults of 101 (for TTree) and 505 (for RNTuple).
- Signatures of the HistoND and HistoNSparseD operations have been changed. Previously, the list of input column names was allowed to contain an extra column for events weights. This was done to align the logic with the THnBase::Fill method. But this signature was inconsistent with all other Histo* operations, which have a separate function argument that represents the column to get the weights from. Thus, HistoND and HistoNSparseD both now have a separate function argument for the weights. The previous signature is still supported, but deprecated: a warning will be raised if the user passes the column name of the weights as an extra element of the list of input column names. In a future version of ROOT this functionality will be removed. From now on, creating a (sparse) N-dim histogram with weights should be done by calling `HistoN[Sparse]D(histoModel, inputColumns, weightColumn)`.
- The string expressions passed to `Vary` calls can now be shortened. If the string begins with '{' and ends with '}' (excluding whitespace, tab and newline characters), RDataFrame will automatically inject the return type in the generated lambda expression before declaring it to the interpreter. This for example allows writing an expression such as `{{px * 0.9, px * 1.1}, {py * 0.9, py * 1.1}}` instead of `ROOT::RVec<ROOT::RVec<ROOT::RVec<float>>>{{px * 0.9, px * 1.1}, {py * 0.9, py * 1.1}}`

## Histograms

A first version of the new histogram package is now available in the `ROOT::Experimental` namespace.
The main user-facing classes are [`ROOT::Experimental::RHist`](https://root.cern/doc/v640/classROOT_1_1Experimental_1_1RHist.html)
and [`ROOT::Experimental::RHistEngine`](https://root.cern/doc/v640/classROOT_1_1Experimental_1_1RHistEngine.html),
as well as the three supported axis types:
[`ROOT::Experimental::RRegularAxis`](https://root.cern/doc/v640/classROOT_1_1Experimental_1_1RRegularAxis.html),
[`ROOT::Experimental::RVariableBinAxis`](https://root.cern/doc/v640/classROOT_1_1Experimental_1_1RVariableBinAxis.html),
[`ROOT::Experimental::RCategoricalAxis`](https://root.cern/doc/v640/classROOT_1_1Experimental_1_1RCategoricalAxis.html).
The two histogram classes are templated on the bin content type and can be multidimensional with dynamic axis types at run-time.
Histograms can be filled via the known `Fill` method, with arguments passed to the variadic function template or as `std::tuple`.
An optional weight must be wrapped in `ROOT::Experimental::RWeight` and passed as the last argument.
Examples are available as [tutorials in the documentation](https://root.cern/doc/v640/group__tutorial__histv7.html).

A key feature of the new histogram package is concurrent filling using atomic instructions.
Support is built into the design of the classes and requires no change of the bin content type:
For `RHistEngine`, directly call one of the `FillAtomic` methods, taking the same arguments as `Fill`.
For `RHist`, first construct an [`ROOT::Experimental::RHistConcurrentFiller`](https://root.cern/doc/v640/classROOT_1_1Experimental_1_1RHistConcurrentFiller.html)
and then create [`RHistFillContext`](https://root.cern/doc/v640/classROOT_1_1Experimental_1_1RHistFillContext.html)s, which will efficiently handle global histogram statistics.

Experimental support for the new histograms is integrated into RDataFrame:
The single overloaded method `Hist()` allows to fill one- and multidimensional histograms.
```C++
auto hist1 = df.Hist(10, {5.0, 15.0}, "x");
auto hist2 = df.Hist({axis1, axis2}, {"x", "y"}, "w");
```

For the moment, histograms can be merged, scaled, sliced, rebinned, and projected.
Furthermore, one-dimensional histograms can be converted to the known `TH1`
via free-standing functions provided in the [`ROOT::Experimental::Hist`](https://root.cern/doc/v640/namespaceROOT_1_1Experimental_1_1Hist.html) namespace.
More operations and conversion for higher dimensions will be added in the future.

Note that all classes are in the `ROOT::Experimental` namespace and remain under active development.
As such, interfaces may change at any moment, without consideration of compatibility.
Nevertheless, we encourage interested parties to test the provided functionality and solicit feedback.
This will allow improving the interfaces and their documentation, and prioritize among missing features.
Once the class layout is finalized, we will also support serialization ("streaming") to ROOT files, which is currently disabled.

## Python Interface

- ROOT dropped support for Python 3.9, meaning ROOT now requires at least Python 3.10.


### Change in memory ownership heuristics

In previous ROOT versions, if a C++ member function took a non-`const` raw pointer, e.g.
```C++
MyClass::add(T* obj)
```
then calling this method from Python on an instance
```Python
my_instance.add(obj)
```
would assume that ownership of `obj` is transferred to `my_instance`.

In practice, many such functions do not actually take ownership.
As a result, this heuristic caused several memory leaks.

Starting with ROOT 6.40, the ROOT Python interfaces no longer assumes ownership transfer for non-`const` raw pointer arguments.

#### What does this mean for you?

Because Python no longer automatically relinquishes ownership, some code that previously relied on the old heuristic may now expose:

  * **Dangling references** on the C++ side
  * **Double deletes**

These issues must now be fixed by managing object lifetimes explicitly.

#### Dangling references

A dangling reference occurs when C++ holds a pointer or reference to an object that has already been deleted on the Python side.

*Example*
```Python
obj = ROOT.MyClass()
my_instance.foo(obj)
del obj  # C++ may still hold a pointer: dangling reference
```

When the Python object is deleted, the memory associated with the C++ object
 is also freed. But the C++ side may want to delete the object as well, for
instance if the destructor of the class of `my_instance` also calls `delete obj`.

*Possible remedies*

 1. **Keep the Python object alive**

    Assign the object to a Python variable that lives at least as long as the C++ reference.

    *Example:* Python lifeline
    ```Python
    obj = ROOT.MyClass()
    my_instance.foo(obj)
    # Setting `obj` as an attribute of `my_instance` makes sure that it's not
    # garbage collected before `my_instance` is deleted:
    my_instance._owned_obj = obj
    del obj  # Reference counter for `obj` doesn't hit zero here
    ```
    Setting the lifeline reference could also be done in a user-side Pythonization of `MyClass::foo`.

 3. **Transfer ownership explicitly to C++**
      * Drop the ownership on the Python side:
        ```Python
        ROOT.SetOwnership(obj, False)
        ```
      * Ensure that the C++ side will take ownership instead, e.g. by explicitly calling `delete` in the class destructor or using smart pointers like `std::unique_ptr`.

 3. **Rely on Pythonizations that imply ownership transfer**

    If the object is stored in a non-owning collection such as a default-constructed `TCollection` (e.g. `TList`), you can make the collection owning before adding any elements:
    ```Python
    coll.SetOwner(True)
    ```
    This will imply ownership transfer to the C++ side when adding elements with `TCollection::Add()`.

#### Note on **TCollection** Pythonization

`TCollection`-derived classes are Pythonized such that when an object is added to an owning collection via `TCollection::Add()`, Python ownership is automatically dropped.

If you identify other cases where such a Pythonization would be beneficial, please report them via a GitHub issue. Users can also implement custom Pythonizations outside ROOT if needed.

#### Double deletes

A double delete indicates that C++ already owns the object, but Python still attempts to delete it.

In this case, you do not need to ensure C++ ownership, as it already exists.
Instead, ensure that Python does not delete the object.

*Possible remedies*

 1. Drop Python ownership explicitly:
    ```Python
    ROOT.SetOwnership(obj, False)
    ```

 2. Pythonize the relevant member function to automatically drop ownership on the Python side (similar to the `TCollection` Pythonization described above).

#### Temporary compatibility option

You can temporarily restore the old heuristic by calling:
```Python
ROOT.SetHeuristicMemoryPolicy(True)
```

after importing ROOT.

This option is intended **for debugging only and will be removed in ROOT 6.44**.

### Drop support for calling C++ functions with non-const pointer references

From now on, we disallow calling C++ functions with non-const pointer references (`T*&`) from Python.
These allow pointer rebinding, which cannot be represented safely in Python and could previously lead to confusing behavior.
A `TypeError` is now raised. Typical ROOT usage is unaffected.

In the rare case where you want to call such a function, please change the C++ interface or - if the interface is outside your control - write a wrapper function that avoids non-const pointer references as arguments.

For example, a function with the signature `bool setPtr(MyClass *&)` could be wrapped by a function that augments the return value with the updated pointer:
```
ROOT.gInterpreter.Declare("""
    std::pair<bool, MyClass*> setPtrWrapper(MyClass *ptr) {
       bool out = setPtr(ptr);
       return {out, ptr};
    }
""")
```
Then, call it from Python as follows:
```Python
# Use tuple unpacking for convenience
_, ptr = cppyy.gbl.setPtrWrapper(ptr)
```

### UHI
#### Backwards incompatible changes
- `TH1.values()` now returns a **read-only** NumPy array by default. Previously it returned a writable array that allowed modifying histogram contents implicitly.
- To modify the histogram buffer, you must now explicitly request it:
```python
h.values(writable=True)[0] = 42
```
- When the histogram storage allows it, `TH1.values()` returns a zero-copy view.
For histogram types that cannot expose their memory layout (`TH*C` and `TProfile*`), `.values()` returns a copy.
In these cases passing `writable=True` is not supported and raises a `TypeError`.

#### New features
- ```TH1.values(flow=True)``` now exposes underflow/ overflow bins when requested.
- ROOT histograms now support **UHI serialization** via intermediate representations with the methods ```_to_uhi_``` and ```_from_uhi_```
- Example usage:
```python
h = ROOT.TH1D("h", "h", 10, -5, 5)
h[...] = np.arange(10)

json_str = json.dumps(h, default=uhi.io.json.default)

h_from_uhi = ROOT.TH1D(json.loads(json_str, , object_hook=uhi.io.json.object_hook))
```

### Removed `TCollection.count()` Pythonization

The Python-only `TCollection.count()` method has been removed. The meaning of the underlying comparison was ambiguous for C++ objects: depending on the element class, it might have counted by matching by value equality or pointer equality. This behavior can vary silently between classes and lead to inconsistent or misleading results, so it was safer to remove the `count()` method.

Users who need to count occurrences in a **TCollection** can explicitly implement the desired comparison logic in Python, for example:

```Python
sum(1 for x in collection if x is obj)   # pointer comparison
sum(1 for x in collection if x == obj)   # value comparison (if defined for the element C++ class)
```

## Machine Learning integration

ROOT provides built-in support for feeding data from ROOT files directly into ML training workflows, without intermediate conversions to other formats.
The central tool for this, previously known as `RBatchGenerator` in the `TMVA` namespace, has been renamed to `ROOT.Experimental.ML.RDataLoader` and redesigned in this release.
Note that the interface and functionality is still experimental: don't rely on it for production purposes, but feedback is much appreciated!
[Tutorials](https://root.cern/doc/master/ml__dataloader__PyTorch_8py.html) are updated to demonstrate the new interface.

### New `RDataLoader` interface
`RDataLoader` is now a Python class that exposes methods for splitting the dataset into training and test sets, and reading batches in 3 formats: NumPy arrays, PyTorch tensors and TensorFlow datasets.
If more formats became popular, they could be easily added in the future.

Usage:
```python
dl = ROOT.Experimental.ML.RDataLoader(
    df,
    batch_size=1024,
    batches_in_memory=10,
    target="label",
)

train, test = dl.train_test_split(test_size=0.2)

for x, y in train.as_torch(): ...
for x, y in test.as_numpy(): ...
for x, y in train.as_tensorflow(): ...
```

A new parameter `batches_in_memory` controls how many batches worth of data are held in the in-memory shuffle buffer at once. Larger values improve shuffling quality at the cost of higher memory usage.

### Cluster-aligned reading and improved shuffling
The loader now reads data in chunks aligned to TTree/RNTuple cluster boundaries on disk rather than fixed-size blocks into an in-memory buffer.
Entries within the buffer are shuffled together before batching if shuffling is enabled. This is done to ensure most efficient I/O scheduling.

### Multiple RDataFrames as input
`RDataLoader` now accepts a list of RDataFrames as input allowing data from multiple files or sources to be combined transparently into a single training stream.


## ROOT executable

- Removed stray linebreak when running `root -q` with input files or commands passed with `-e`.
  This ensures that there is no superfluous output when running `root`.
  Note that ROOT 6.38 already removed a spurious newline when starting `root` **without** input files or commands.
- The ROOT header printed by default when starting the interpreter is now
  suppressed when scripts or commands are passed on the command line.
  This makes sure the output doesn't contain unexpected and ROOT
  version dependent output, which helps in case the output is piped to other
  commands or compared to reference files.

## Command-line utilities

- `rootls` has a new flag: `-c / --allCycles`, which displays all cycles of each object in the inspected file/directory even in normal mode (cycles are already displayed by default with `-l` or `-t`).
- `rootcp` and `rootrm` have a new native implementation, which should make them significantly faster to startup and usable even without Python.

## JavaScript ROOT

## Experimental features

### Opting out of object auto registration
In preparation for ROOT 7, ROOT 6.40 introduces an experimental mode for opting out of the auto-registration of objects.
In ROOT 7, this will be the default, and an opt-in will be required to make these objects auto-register themselves.
The table below shows which objects currently honour this mode, and which objects are planned to be added on the path to ROOT 7.

The planned ROOT 7 behaviour can be enabled in one of three ways:
1. `ROOT::Experimental::DisableObjectAutoRegistration()`: This disables auto registration *for the current thread*.
2. Setting the environment variable `ROOT_OBJECT_AUTO_REGISTRATION=0`: This sets the default for every thread that starts.
3. In `.rootrc`, set the entry `Root.ObjectAutoRegistration: 0`: This sets the default for every thread that starts.

**Note that method 1 affects only the current thread**, whereas methods 2 and 3 set the default for every thread that is started in this ROOT session.
Using `ROOT::Experimental::EnableObjectAutoRegistration()`, the auto-registration can be enabled for a single thread without affecting the rest of the session.

Consult the doxygen documentation of these functions in the [ROOT::Experimental namespace](https://root.cern.ch/doc/v640/namespaceROOT_1_1Experimental.html) for details.

|                       | Honours `DisableObjectAutoRegistration()`? | Could this be disabled previously? |
| --------------------- | ------------------------------------------ | ---------------------------------- |
| TH1 and derived       | Yes                                        | TH1::AddDirectoryStatus()          |
| TGraph2D              | Yes                                        | TH1::AddDirectoryStatus()          |
| RooPlot               | Yes                                        | RooPlot::addDirectoryStatus()      |
| TEfficiency           | Yes                                        | No                                 |
| TProfile2D            | Yes                                        | TH1::AddDirectoryStatus()          |
| TEntryList            | No, planned for 6.42                       | No                                 |
| TEventList            | No, planned for 6.42                       | No                                 |
| TFunction             | No, work in progress                       | No                                 |

## Versions of built-in packages

The version of the following packages has been updated:

 - cppzeromq: 4.10.0
 - fftw3: 3.3.10
 - freetype: 2.14.3
 - gsl: 2.8
 - gtest: 1.17.0
 - giflib: 5.2.2
 - libjpeg-turbo 3.1.3 is now used as a replacement for libjpeg
 - libpng: 1.6.4
 - libzeromq: 4.3.5
 - lz4: 1.10.0
 - lzma: 5.8.2
 - xrootd: 5.9.2
 - zlib: 1.3.2
 - zstd: 1.5.7


## Items addressed for this release
More than 130 items were addressed for this release:

  * [[#21997](https://github.com/root-project/root/issues/21997)] - Memory leak in TFile::Open with UPDATE mode
  * [[#21991](https://github.com/root-project/root/issues/21991)] - Link errors for tbb and xxhash libraries
  * [[#21974](https://github.com/root-project/root/issues/21974)] - [ntuple] Incorrect treatment of I/O rule with empty source
  * [[#21933](https://github.com/root-project/root/issues/21933)] - Web canvas saves files in the wrong format if extension is not specified
  * [[#21915](https://github.com/root-project/root/issues/21915)] - [xrootd] error when building: XROOTD_CLIENT_LIBRARIES-NOTFOUND
  * [[#21881](https://github.com/root-project/root/issues/21881)] - [gui,asimage] crash when opening xpm icon in batch mode
  * [[#21880](https://github.com/root-project/root/issues/21880)] - [asimage] crash with DrawCircle
  * [[#21863](https://github.com/root-project/root/issues/21863)] - hadd does not perform type checks of objects to be merged
  * [[#21787](https://github.com/root-project/root/issues/21787)] - ROOT and Python 3.15
  * [[#21782](https://github.com/root-project/root/issues/21782)] - [ML] ROOT.Experimental.ML.CreatePyTorchGenerators only uses the first dataframe
  * [[#21774](https://github.com/root-project/root/issues/21774)] - [tmva][sofie] Pool operator ignores ceil_mode attribute, always uses floor division for output shape
  * [[#21759](https://github.com/root-project/root/issues/21759)] - [tmva][sofie] ParseBatchNormalization accepts invalid input count and fails late
  * [[#21758](https://github.com/root-project/root/issues/21758)] - [RDF] "error: macro name must be an identifier" when declaring C++ code for a distributed RDF
  * [[#21756](https://github.com/root-project/root/issues/21756)] - [RDF] RDF.RSnapshotOptions with a distributed RDF
  * [[#21747](https://github.com/root-project/root/issues/21747)] - [io][ML] Sporadic crash of the RDataLoader (and RBatchGenerator)
  * [[#21736](https://github.com/root-project/root/issues/21736)] - [tmva][sofie] ParseMod validation for float/double never fires
  * [[#21734](https://github.com/root-project/root/issues/21734)] - [tmva][sofie] Mod/FMod constant-folding uses std::pow instead of modulo
  * [[#21732](https://github.com/root-project/root/issues/21732)] - invalid looping over the `std::array<long double,..>` objects  in PyROOT
  * [[#21717](https://github.com/root-project/root/issues/21717)] - RooFit::chi2FitTo ignores the Range argument
  * [[#21716](https://github.com/root-project/root/issues/21716)] - Strange error message when reading RNTuples
  * [[#21699](https://github.com/root-project/root/issues/21699)] - Missing documentation for RooChi2Var since ROOT 6.36
  * [[#21693](https://github.com/root-project/root/issues/21693)] - [PyROOT] ROOT package import method changes gDirectory.pwd() result
  * [[#21682](https://github.com/root-project/root/issues/21682)] - [tmva][sofie] ReduceMean: loop variable typo in kFirst path causes infinite loop or wrong result
  * [[#21667](https://github.com/root-project/root/issues/21667)] - Missing alignment information in TClass/TStreamerInfo
  * [[#21608](https://github.com/root-project/root/issues/21608)] - Position dependent code of builtin lzma breaks ATLAS stat utils
  * [[#21576](https://github.com/root-project/root/issues/21576)] - Add public accessor to RooPlot::_hist ?
  * [[#21561](https://github.com/root-project/root/issues/21561)] - [graf2d] Cannot create gif images on some platforms
  * [[#21542](https://github.com/root-project/root/issues/21542)] - [RF] Behavior of RooRealVar::removeRange
  * [[#21541](https://github.com/root-project/root/issues/21541)] - [RF] `createChi2` ignores ranges provided by the Range() option
  * [[#21539](https://github.com/root-project/root/issues/21539)] - [tmva][sofie] `ROperator_Elu.hxx` generates incorrect C++ inference code when alpha != 1.0.
  * [[#21484](https://github.com/root-project/root/issues/21484)] - Last axis title is not set for THn and THnSparseD histograms, affects also HistoND and HistoNSparseD
  * [[#21469](https://github.com/root-project/root/issues/21469)] - TPad::PaintHatches() fails for complex polygons
  * [[#21438](https://github.com/root-project/root/issues/21438)] - Poisson interval calculation with asymptotic approximation
  * [[#21405](https://github.com/root-project/root/issues/21405)] - [PyROOT] Assinging to `TemplateProxy` affects others
  * [[#21379](https://github.com/root-project/root/issues/21379)] - Missing numpy.uint8 pythonisation in SetBranchAddress in ROOT 6.38
  * [[#21378](https://github.com/root-project/root/issues/21378)] - [Python] NumPy conversion fails for multidimensional arrays (incorrect itemsize in LowLevelView)
  * [[#21366](https://github.com/root-project/root/issues/21366)] - TGraph2D::Build function may leave the newly created object in inconsistent non-initialised state
  * [[#21317](https://github.com/root-project/root/issues/21317)] - New feature: Add modified Anderson Bjork's algorithm for solving nonlinear equations
  * [[#21284](https://github.com/root-project/root/issues/21284)] - [RF] Add new PDF Student-T distribution as one of RooAbsPdfs
  * [[#21253](https://github.com/root-project/root/issues/21253)] - TH1::GetSumOfAllWeights return 9xvalue for 1D and (3x) 2D histograms when including underflows/overflows
  * [[#21244](https://github.com/root-project/root/issues/21244)] - TPad Print() Title option issue
  * [[#21173](https://github.com/root-project/root/issues/21173)] - TLegend header drawn in different position with jsroot and more than 1 column
  * [[#21165](https://github.com/root-project/root/issues/21165)] - RDataFrame progress bar changes stream output precision
  * [[#21159](https://github.com/root-project/root/issues/21159)] - [RF] Unexpected result with RooFFTConvPDF after ROOT 6.32
  * [[#21104](https://github.com/root-project/root/issues/21104)] - Out-of-bounds access in TFormula constructor
  * [[#21098](https://github.com/root-project/root/issues/21098)] - Typo in RooWorkspace factory documentation
  * [[#21080](https://github.com/root-project/root/issues/21080)] - Bad interference for TH1.Fit and EnableImplicitMT
  * [[#21070](https://github.com/root-project/root/issues/21070)] - RedirectServers - Server not redirected
  * [[#21066](https://github.com/root-project/root/issues/21066)] - [RF] Crash doing HistFactory Workspace combination
  * [[#21058](https://github.com/root-project/root/issues/21058)] - ROOT interpreter crashes when using `nlohmann::json`
  * [[#20973](https://github.com/root-project/root/issues/20973)] - TEveManager::GetGeometry could support ROOT file with non standard extension
  * [[#20954](https://github.com/root-project/root/issues/20954)] - [Minuit2] Analytical Hessian indexing mismatch
  * [[#20948](https://github.com/root-project/root/issues/20948)] - [net] Insufficient length validation in TAuthenticate::SecureRecv()
  * [[#20913](https://github.com/root-project/root/issues/20913)] - MIGRAD can call G2 even if HasG2 is set to false
  * [[#20904](https://github.com/root-project/root/issues/20904)] - [RF] RooWorkspace::data returns null pointer if the workspace contains 52 or more datasets
  * [[#20894](https://github.com/root-project/root/issues/20894)] - ProfileLikelihood calculator defines wrongly the tolerance
  * [[#20872](https://github.com/root-project/root/issues/20872)] - hadd crashes when already existing output file is used with remote file input paths
  * [[#20860](https://github.com/root-project/root/issues/20860)] - Inconsistent information for histogram contours
  * [[#20858](https://github.com/root-project/root/issues/20858)] - Make "LIST" draw option for contour plot work with "CONT1", "CONT2", and "CONT3" options
  * [[#20846](https://github.com/root-project/root/issues/20846)] - [RDF] Implement `Display` for distributed execution
  * [[#20834](https://github.com/root-project/root/issues/20834)] - c++20 related test filure on ix86 (32 bit intel)
  * [[#20831](https://github.com/root-project/root/issues/20831)] - New test failure with gcc 16 (possibly c++20 related)
  * [[#20816](https://github.com/root-project/root/issues/20816)] - Improve `RInterface::HistoND` signature with an extra argument for weight
  * [[#20792](https://github.com/root-project/root/issues/20792)] - TGraph.Fit() usage with EnableImplicitMT() produces inconsistent results
  * [[#20790](https://github.com/root-project/root/issues/20790)] - Applications compiled on MacOS with -Wshadow report warnings from 14 ROOT headers
  * [[#20786](https://github.com/root-project/root/issues/20786)] - thisroot.sh can set incorrect ROOTSYS
  * [[#20780](https://github.com/root-project/root/issues/20780)] - Font precision 3 doesn't work with divided canvas
  * [[#20761](https://github.com/root-project/root/issues/20761)] - Chi2Test: Report residuals also for higher dimensions
  * [[#20755](https://github.com/root-project/root/issues/20755)] - `minimal` should be used only to set default feature
  * [[#20750](https://github.com/root-project/root/issues/20750)] - Insufficient checks around R__unzip_header
  * [[#20743](https://github.com/root-project/root/issues/20743)] - [ci] PR branch not checked out in CI tests if it's called `master`
  * [[#20738](https://github.com/root-project/root/issues/20738)] - roottest-cling-parsing-semicolon  is not running, because semicolon is special in cmake?
  * [[#20733](https://github.com/root-project/root/issues/20733)] - [math] xyexpo does not work in TFormula
  * [[#20719](https://github.com/root-project/root/issues/20719)] - [math] TH2 FillRandom xygaus does not work
  * [[#20706](https://github.com/root-project/root/issues/20706)] - Segmentation violation in the TTree::Merge method for root >= v6.36.00
  * [[#20703](https://github.com/root-project/root/issues/20703)] - ROOT::Fit::FitResult fixed / bound parameter states not updated
  * [[#20688](https://github.com/root-project/root/issues/20688)] - Make `TF3::Draw()` behave as expected
  * [[#20687](https://github.com/root-project/root/issues/20687)] - Zero-parameter `TF1` created from C++ functor crashes `TCanvas::SaveAs()`
  * [[#20674](https://github.com/root-project/root/issues/20674)] - .rootlogon.C ignored when running TRootBrowser
  * [[#20665](https://github.com/root-project/root/issues/20665)] - Analytical Hessian is not made positive-definite during minimizer seeding in Minuit2
  * [[#20602](https://github.com/root-project/root/issues/20602)] - TGTextEntry spills text out of the text box on macOS but not on Ubuntu
  * [[#20572](https://github.com/root-project/root/issues/20572)] - A public field with a "private" type in the TFileMergeInfo class
  * [[#20511](https://github.com/root-project/root/issues/20511)] - CMSSW DQM bin-by-bin comparison failed with ROOT 6.36
  * [[#20506](https://github.com/root-project/root/issues/20506)] - [RDF] When combining a JIT-ted Vary and Define, the varied values are missing in a Snapshot
  * [[#20476](https://github.com/root-project/root/issues/20476)] - Save THStack in root file
  * [[#20472](https://github.com/root-project/root/issues/20472)] - jsroot from pip install root not working in notebooks (no TWebCanvas?)
  * [[#20470](https://github.com/root-project/root/issues/20470)] - ROOT still uses hard-coded open mode flags, overriding umask and ACL settings
  * [[#20448](https://github.com/root-project/root/issues/20448)] - [ntuple] detect custom streamers of class members
  * [[#20436](https://github.com/root-project/root/issues/20436)] - [hist] TScatter (unwanted) background color is at 10 instead of at 0 if negative Z content
  * [[#20383](https://github.com/root-project/root/issues/20383)] - [RF] Broken rf501_simultaneouspdf.py  in dev3/nighlies
  * [[#20377](https://github.com/root-project/root/issues/20377)] - Investigate I/O of `std::unique_ptr` on latest MacOS beta
  * [[#20356](https://github.com/root-project/root/issues/20356)] - Auxiliaries of volumes are not exportet in gdml
  * [[#20315](https://github.com/root-project/root/issues/20315)] - Some test fails with a minimal build
  * [[#20312](https://github.com/root-project/root/issues/20312)] - Support for std::complex on Windows
  * [[#20282](https://github.com/root-project/root/issues/20282)] - [ntuple] use meta normalized name for streamer info records
  * [[#20265](https://github.com/root-project/root/issues/20265)] - UHI histogram kind
  * [[#20208](https://github.com/root-project/root/issues/20208)] - Flag to disable cling's `NVPTX` target
  * [[#20174](https://github.com/root-project/root/issues/20174)] - TH1 and derivates projections produce incorrect results for histograms with infinite bin edges
  * [[#20085](https://github.com/root-project/root/issues/20085)] - [cppyy] Review xfail'ed tests
  * [[#20066](https://github.com/root-project/root/issues/20066)] - [Python] cppyy cannot bind select types
  * [[#20047](https://github.com/root-project/root/issues/20047)] - Code coverage shows incorrect missed lines
  * [[#19798](https://github.com/root-project/root/issues/19798)] - Removed option `tbb` is still considered to enable secret optimization of `ROOT::gCoreMutex`
  * [[#19371](https://github.com/root-project/root/issues/19371)] - "read too few bytes" with class version zero for 'foreign' classes.
  * [[#19329](https://github.com/root-project/root/issues/19329)] - h2root fragilities
  * [[#18949](https://github.com/root-project/root/issues/18949)] - `Vc.pcm` not found when running root after building with external Vc with runtime_cxxmodules ON
  * [[#18837](https://github.com/root-project/root/issues/18837)] - [Python] Calling `std::span::begin()` broken with gcc15
  * [[#18754](https://github.com/root-project/root/issues/18754)] - Replace use of TSystem::ConcatFileName with PrependPathName.
  * [[#18718](https://github.com/root-project/root/issues/18718)] - [RF] Regression in ranged fits with RooSimultaneous due to new likelihood evaluation
  * [[#18471](https://github.com/root-project/root/issues/18471)] - Migrate from GLEW to libepoxy or GLAD
  * [[#18314](https://github.com/root-project/root/issues/18314)] - [ntuple] Heuristically reduce memory usage of buffered writing
  * [[#17697](https://github.com/root-project/root/issues/17697)] - pyroot can trigger clang assertion failure on enum with value too big to fit in int64_t
  * [[#17429](https://github.com/root-project/root/issues/17429)] - [RF] Inconsistent  prints from some RooStats classes
  * [[#17019](https://github.com/root-project/root/issues/17019)] - [reve] Invalid bounding box calculation for REvePointSet and REveBoxSet
  * [[#16673](https://github.com/root-project/root/issues/16673)] - [RF] Different behaviour of multi-range fit in RooAddPdf and RooProdPdf
  * [[#16601](https://github.com/root-project/root/issues/16601)] - [6.32/master] Fail to materialise symbols from `libMatrix`
  * [[#16512](https://github.com/root-project/root/issues/16512)] - Enable embedding hyperlinks in TLegend
  * [[#15872](https://github.com/root-project/root/issues/15872)] - [PyROOT] The cppyy version inside ROOT doesn't support `long long`
  * [[#15792](https://github.com/root-project/root/issues/15792)] - root_generate_dictionary should check target OUTPUT_NAME
  * [[#15783](https://github.com/root-project/root/issues/15783)] - [ntuple, DAOS] maximum cage size should be part of the anchor
  * [[#15537](https://github.com/root-project/root/issues/15537)] - [cling] Crash when non-void function does not return a value
  * [[#15520](https://github.com/root-project/root/issues/15520)] - Rewrite the RDataFrame JIT logic to avoid controlled leaks
  * [[#15474](https://github.com/root-project/root/issues/15474)] - [Web Graphics] Two failures with one simple PyROOT plotter
  * [[#14943](https://github.com/root-project/root/issues/14943)] - (Web) graphics does not work from PyROOT
  * [[#14790](https://github.com/root-project/root/issues/14790)] - [DF] 1D `std::array` "has both a leaf count and a static length"
  * [[#14446](https://github.com/root-project/root/issues/14446)] - gtest-math-matrix-test-testMatrixTSparse fails on aarch64, ppc64le and s390x with gcc 14
  * [[#14209](https://github.com/root-project/root/issues/14209)] - [cling] Enable JITLink for AArch64 and x86_64 on Linux
  * [[#13744](https://github.com/root-project/root/issues/13744)] - [PyROOT, Graphics] Cannot pause pyROOT script with `input()` and see canvases
  * [[#11768](https://github.com/root-project/root/issues/11768)] - [RF] Error out in `RooStats::sPlot` if variable for sWeights is already used in the fit
  * [[#10678](https://github.com/root-project/root/issues/10678)] - TH3::Interpolate min/max value condition behaves differently for TH1/TH2
  * [[#10254](https://github.com/root-project/root/issues/10254)] - [TTreeReader] No diagnostics printed before crash when reading a type with no dictionaries
  * [[#9354](https://github.com/root-project/root/issues/9354)] - [Win] Warnings about missing PCMs
  * [[#8875](https://github.com/root-project/root/issues/8875)] - [PyROOT] Error when copying a tuple into a specific position of a vector of tuples in PyROOT
  * [[#8582](https://github.com/root-project/root/issues/8582)] - TThreadTimer behavior
  * [[#8199](https://github.com/root-project/root/issues/8199)] - ROOT::Math C++ PRBS generator
  * [[#8125](https://github.com/root-project/root/issues/8125)] - BMP TASImage artefacts and crashes
  * [[#7196](https://github.com/root-project/root/issues/7196)] - gSystem->GetMemInfo reports wrong used RAM memory
  * [[#7167](https://github.com/root-project/root/issues/7167)] - [TTreeProcessorMP] Warn if Process is called from a multi-thread program
  * [[#7052](https://github.com/root-project/root/issues/7052)] - rootcp --replace option ineffective
  * [[ROOT-10972](https://its.cern.ch/jira/browse/ROOT-10972)] - [TTreePerfStats] No result with EnableImplicitMT
  * [[ROOT-10808](https://its.cern.ch/jira/browse/ROOT-10808)] - Crash in TPoolManager's destructor
  * [[ROOT-10778](https://its.cern.ch/jira/browse/ROOT-10778)] - When using copyTree with TChain , friend Tree's are not correctly copied
  * [[ROOT-10728](https://its.cern.ch/jira/browse/ROOT-10728)] - TClass::CanSplit() should automatically return 0 if a base class has a custom streamer.
  * [[ROOT-10614](https://its.cern.ch/jira/browse/ROOT-10614)] - TFormula tests fail when vectorization is enabled
  * [[ROOT-10567](https://its.cern.ch/jira/browse/ROOT-10567)] - Fix consistency in computing number of entries after performing additions/subtractions of TH1 objects
  * [[ROOT-9971](https://its.cern.ch/jira/browse/ROOT-9971)] - VisualizeError seems to ignore the normalisation of extended PDFs
  * [[ROOT-8842](https://its.cern.ch/jira/browse/ROOT-8842)] - TTreeReaderFast crashes reading flat TTree (fork: bbockelm/root branch: root-bulkapi-fastread-v2)
  * [[ROOT-7499](https://its.cern.ch/jira/browse/ROOT-7499)] - ExpectedData generated from RooSimultaneous does not have non-integer weights
  * [[ROOT-5306](https://its.cern.ch/jira/browse/ROOT-5306)] - Read a file with a versioned class layout fails if the current class layout is unversioned
  * [[ROOT-5174](https://its.cern.ch/jira/browse/ROOT-5174)] - rootcling without linkdef

