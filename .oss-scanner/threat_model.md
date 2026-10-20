# Threat model

## What ROOT is and where input enters

ROOT is a scientific framework and python package for the processing, analysis and visualisation of scientific data. It is born for High Energy Physics (HEP) and used in academia and industry.
Three of its most important features are:
 - The ability to read and write data in ROOT files. Data can be individual objects or columnar datasets (`TTree`and `RNTuple`).
 - A C++ interpreter, `Cling` based on Clang: code can be JITed at runtime
 - A Python interface made of dynamically created Python bindings

ROOT can be used via Python programs, or C++ programs, as a library.
ROOT files, as well as interpreted "C++ Macros" (JITted programs) are given in input to ROOT.

## How to exercise it

* Rebuild with `make -j8` after your changes.
* Test with `ctest -j8`. Some tests will fail since they require network.

## How we would like reports and patches to look

* A reproducer as a small python or C++ program against the public API, a `make test`
  recipe; plus the exact commit, configure options and the sanitizer output.
* The affected `master`/`v6-*-00-patches` release branches if known, since
  we fix all supported versions.
* Minimal patches against `master` following `CONTRIBUTING.md`, ideally with a
  regression test.