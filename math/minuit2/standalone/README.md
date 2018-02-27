This is the Minuit2 fitter standalone edition, from the [ROOT] toolkit. It uses [CMake] 3.1+ to build.
For information about the Minuit2 fitter, please see the [documentation in ROOT][minuitdoc].
See `DEVELOP.md` for information about extracting from [ROOT].

## Building

To build, use the standard [CMake] procedure; on most systems, this looks like:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

Of course, GUIs, IDEs, etc. that work with CMake will work with this package.

The standard CMake variables, such as `CMAKE_BUILD_TYPE` and `CMAKE_INSTALL_PREFIX`, work with Minuit2.  There are two other options:

* `MINUIT2_MPI` activates the (outdated C++) MPI bindings.
* `MINUIT2_OMP` activates OpenMP (make sure all FCNs are threadsafe).

## Installing or using in another package

You can install the package using `cmake --build --target install .` (or `make install` if directly using the make system), or you can use it from the build directory. You can also include it in another CMake project using `add_subdirectory()` and linking to the `Minuit2` target. Since this package also exports targets, `find_package(Minuit2)` will also work once this package is built or installed. (For the curious, CMake adds a config script to `~/.cmake/packages` when building or
`$CMAKE_INSTALL_PREFIX/share/cmake/Modules` when installing a package that has export commands.)

To repeat; using this in your own CMake project usually amounts to:

```cmake
find_package(Minuit2)
# OR
add_subdirectory(Minuit2)

target_link_libraries(MyExeOrLib PUBLIC Minuit2)
```

You do not need to add include directories or anything else for Minuit2; the CMake target system handles all of this for you.

## Packaging

Minuit2 also has basic support for CPack to make installers for different platforms. To build a source package:

```bash
make source_package
```

To build a binary package (add other generators with `-G`):
```bash
make package
```


[ROOT]: https://root.cern.ch
[minuitdoc]: https://root.cern.ch/root/htmldoc/guides/users-guide/ROOTUsersGuide.html#minuit2-package
[CMake]: https://cmake.org
