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
  - llvm19

## Introspection

-

## Just-in-Time Compilation

-

## Incremental C++

-

## Misc

-

## Fixed Bugs

[XXX](https://github.com/compiler-research/CppInterOp/issues/XXX)

<!---Get release bugs
 git log v1.6.0..main | grep 'Fixes|Closes'
 --->

## Special Kudos

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

A B (N)

<!---Find contributor list for this release
 git log --pretty=format:"%an"  v1.6.0...main | sort | uniq -c | sort -rn |\
   sed -E 's,^ *([0-9]+) (.*)$,\2 \(\1\),'
--->
