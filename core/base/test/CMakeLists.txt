# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

ROOT_ADD_GTEST(CoreBaseTests
  TNamedTests.cxx
  TQObjectTests.cxx
  TExceptionHandlerTests.cxx
  TStringTest.cxx
  TUrlTest.cxx
  TBitsTests.cxx
  TRefTests.cxx
  TROOTTests.cxx
  LIBRARIES RIO Core)

ROOT_ADD_GTEST(CoreErrorTests TErrorTests.cxx LIBRARIES Core)

ROOT_ADD_GTEST(CoreSystemTests TSystemTests.cxx LIBRARIES Core)

ROOT_ADD_GTEST(CoreColorTests TColorTests.cxx LIBRARIES Core)

configure_file(Foo.C Foo.C COPYONLY)
ROOT_ADD_GTEST(IncludePathTest IncludePathTest.cxx LIBRARIES Core)
