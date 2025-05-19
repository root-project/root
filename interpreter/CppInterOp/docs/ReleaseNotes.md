# Introduction

This document contains the release notes for the language interoperability
library CppInterOp, release 1.7.0. CppInterOp is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org%3E) compiler
infrastructure. Here we describe the status of CppInterOp in some detail,
including major improvements from the previous release and new feature work.
Note that if you are reading this file from a git checkout, this document
applies to the *next* release, not the current one.

CppInterOp exposes API from Clang and LLVM in a backward compatibe way. The API
support downstream tools that utilize interactive C++ by using the compiler as
a service. That is, embed Clang and LLVM as a libraries in their codebases. The
API are designed to be minimalistic and aid non-trivial tasks such as language
interoperability on the fly. In such scenarios CppInterOp can be used to provide
the necessary introspection information to the other side helping the language
cross talk.

## What's New in CppInterOp 1.7.0?

Some of the major new features and improvements to CppInterOp are listed here.
Generic improvements to CppInterOp as a whole or to its underlying
infrastructure are described first.

## External Dependencies

- CppInterOp now works with:
  - llvm20

## Introspection

- Added `BestOverloadFunctionMatch` and `IsFunction`; removed
  `BestTemplateFunctionMatch`.
- Enhanced overload resolution and template instantiation capabilities.
- Improvements to function signature support for `FunctionTemplateDecl`s.
- Extended support for `GetClassTemplatedMethods`, `LookupConstructor`, and
  better handling in `IsConstructor`.

## Incremental C++

- Improved error propagation in interpreter creation.
- Added undo/unload features with REPL support for the Cling backend.
- Enhancements in interpreter argument handling.

## Misc

- Fixed symbol visibility in the C API.
- Fixed symbol visibility issues in the C API.
- Improved CI and Emscripten build system including browser testing support.
- Updated build compatibility with Cling v1.2 and LLVM 20.
- Improved support and tests for Emscripten builds.
- Enabled shared object loading tests in Emscripten.
- Added automated coverage jobs and various test enhancements.
- Refined wrapper generation and fixed indentation consistency.


## Special Kudos

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

A B (N)
mcbarton (30)
Aaron Jomy (15)
Anutosh Bhat (6)
Gnimuc (5)
Vipul Cariappa (3)
Vassil Vassilev (2)
Abhinav Kumar (2)
Yupei Qi (1)
jeaye (1)
