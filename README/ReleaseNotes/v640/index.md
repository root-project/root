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

## Deprecation and Removal

* The `TH1K` class was removed. `TMath::KNNDensity` can be used in its stead.
* The headers in `RooStats/HistFactory` for data classes related to the measurement definition were merged into the `RooStats/HistFactory/Measurement.h` header to simplify usage and development. For now, the whole set of header files is kept for backwards compatibility, but the empty headers will be removed in ROOT 7.
* The `TObject` equality operator pythonization (`TObject.__eq__`) that was deprecated in ROOT 6.38 and scheduled for removal in ROOT 6.40 is removed.
* Comparing C++ `nullptr` objects with `None` in Python now raises a `TypeError`, as announced in the ROOT 6.38 release notes. Use truth-value checks like `if not x` or `x is None` instead.
* The `TGLIncludes.h` and `TGLWSIncludes.h` that were deprecated in ROOT 6.38 and scheduled for removal are gone now. Please include your required headers like `<GL/gl.h>` or `<GL/glu.h>` directly.
* The GLEW headers (`GL/eglew.h`, `GL/glew.h`, `GL/glxew.h`, and `GL/wglew.h`) that were installed when building ROOT with `builtin_glew=ON` are no longer installed. This is done because ROOT is moving away from GLEW for loading OpenGL extensions.

## Build System

## Core Libraries

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

## Command-line utilities

## JavaScript ROOT

## Experimental features

## Versions of built-in packages

## Items addressed for this release
