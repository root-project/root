# RooBatchCompute

`libRooBatchCompute` provides the vectorized/parallelized computation kernels
that RooFit uses to evaluate its models. The actual kernels do not live in
`libRooBatchCompute` itself: they are compiled into one *backend* library per
target architecture, and `libRooBatchCompute` loads the appropriate one at
runtime with `gSystem->Load()`:

| Backend library                | Loaded by                          |
|--------------------------------|------------------------------------|
| `libRooBatchCompute_GENERIC`   | `RooBatchCompute::initCPU()`       |
| `libRooBatchCompute_SSE4.1`    | `RooBatchCompute::initCPU()`       |
| `libRooBatchCompute_AVX`       | `RooBatchCompute::initCPU()`       |
| `libRooBatchCompute_AVX2`      | `RooBatchCompute::initCPU()`       |
| `libRooBatchCompute_AVX512`    | `RooBatchCompute::initCPU()`       |
| `libRooBatchCompute_CUDA`      | `RooBatchCompute::initCUDA()`      |

Because the backends are plugins that are looked up by name in ROOT's dynamic
library search path, they do not have to be built together with the rest of
ROOT. This is in particular useful for the CUDA backend: it is the only part of
ROOT that needs the CUDA toolkit, and distributions may not want to make all of
ROOT depend on it.

## Building only the CUDA backend against an installed ROOT

The `CMakeLists.txt` in this directory doubles as the top-level `CMakeLists.txt`
of a standalone project that builds nothing but `libRooBatchCompute_CUDA`. Point
CMake at this directory instead of at the top of the ROOT source tree:

```bash
cmake -S <root-source-dir>/roofit/batchcompute -B rbc_cuda_build \
      -DCMAKE_PREFIX_PATH=<root-install-prefix> \
      -DCMAKE_CUDA_ARCHITECTURES=<architectures>
cmake --build rbc_cuda_build -j$(nproc)
cmake --install rbc_cuda_build
```

Notes:

* The ROOT sources have to be the ones the installed ROOT was built from. The
  backend and `libRooBatchCompute` share the `RooBatchComputeInterface` ABI,
  which is not stable across ROOT versions.
* `CMAKE_PREFIX_PATH` can be omitted if `thisroot.sh` has been sourced. Instead
  of the installation prefix, `ROOT_DIR` can be pointed directly at the
  directory that contains `ROOTConfig.cmake` (`<root-install-prefix>/cmake` in
  ROOT's default install layout, but distributions often move it). Note that a
  `ROOT_DIR` that does not exist is silently ignored by `find_package()`, which
  then falls back to any other ROOT it can find, for instance one that is in
  `PATH`; check the `Building libRooBatchCompute_CUDA standalone against ROOT
  ...` line that the configuration step prints.
* `CMAKE_CUDA_ARCHITECTURES` can be omitted, in which case CMake compiles for
  whatever architecture the CUDA compiler defaults to, just like the regular
  ROOT build does.
* No build type is chosen for you. Configure with `-DCMAKE_BUILD_TYPE=Release`
  for an optimized build like the one the ROOT build does by default, or pass
  your own optimization flags in `CMAKE_CXX_FLAGS` and `CMAKE_CUDA_FLAGS`. If
  neither is given, the kernels are compiled without any optimization flags and
  the configuration step warns about it.
* The build only needs a CUDA compiler, the ROOT headers and `libRooBatchCompute`
  from the ROOT installation. No dictionaries are generated and no part of ROOT
  is rebuilt.
* By default the library is installed straight into the library directory of the
  ROOT installation that was found (`${ROOT_LIBRARY_DIR}`), which is where
  `RooBatchCompute::initCUDA()` will look for it. To stage it somewhere else,
  for example when building a package, set
  `-DRooBatchCompute_CUDA_INSTALL_LIBDIR=lib` — a relative path is interpreted
  with respect to `CMAKE_INSTALL_PREFIX`, so the usual
  `CMAKE_INSTALL_PREFIX`/`DESTDIR` mechanisms apply. Wherever the library ends
  up, that directory has to be in ROOT's dynamic library search path (i.e. in
  `$ROOTSYS/lib`, in `LD_LIBRARY_PATH`, or added to `Unix.*.Root.DynamicPath` in
  `.rootrc`).

The ROOT installation itself does not need to know anything about the CUDA
backend: it is enough that `libRooBatchCompute_CUDA` is findable. RooFit only
attempts to load it when a computation is requested on the CUDA backend, so a
ROOT installation without the library keeps working as before.

## Building everything at once

Nothing changes for the regular ROOT build: configuring ROOT with `-Dcuda=ON`
builds `libRooBatchCompute_CUDA` alongside the CPU backends, exactly as before.
