% ROOT Version 6.40 Release Notes
% 2026-5
<a name="TopOfPage"></a>

## Introduction

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

## Deprecations

* The headers in `RooStats/HistFactory` for data classes related to the measurement definition were merged into the `RooStats/HistFactory/Measurement.h` header to simplify usage and development. For now, the whole set of header files is kept for backwards compatibility, but the empty headers will be removed in ROOT 7.
* The `TROOT::GetSourceDir()` method is deprecated and will be removed in ROOT 6.42. It stopped making sense because the ROOT source is generally not shipped alongside ROOT in the `src/` subdirectory anymore.
* Using the `rpath` build option - deprecated and without effect since ROOT 6.38 - is now scheduled to give configuration errors starting from ROOT 6.42.

## Removals

* The `TH1K` class was removed. `TMath::KNNDensity` can be used in its stead.
* The `TObject` equality operator pythonization (`TObject.__eq__`) that was deprecated in ROOT 6.38 and scheduled for removal in ROOT 6.40 is removed.
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

## Build System

## Core Libraries
- ROOT now adds a RUNPATH to compiled macros. This ensures that when compiled macros are loaded, they load the libraries that belong to the ROOT installation
  that compiled the macro. See [TSystem::SetMakeSharedLib()](https://root.cern.ch/doc/master/classTSystem.html#a80cd12e064e2285b35e9f39b5111d20e) for
  customising or disabling the RUNPATH.

## Geometry

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

* `RRawPtrWriteEntry` is now part of the stable API, in the `ROOT::Detail` namespace. It is useful for frameworks passing data to RNTuple as `const` raw pointers.

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

## RooFit

## RDataFrame

- The message shown in ROOT 6.38 to inform users about change of default compression setting used by Snapshot (was 101 before 6.38, became 505 in 6.38) is now removed.

## Python Interface

ROOT dropped support for Python 3.9, meaning ROOT now requires at least Python 3.10.

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

## JavaScript ROOT

## Experimental features

## Versions of built-in packages

## Items addressed for this release
