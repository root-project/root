# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# FIXME: The tests in core should require only libCore. OTOH, TQObjectTests uses the interpreter to register the class.
# This means that if we run make CoreBaseTests the executable wouldn't be runnable because it requires libCling and
# onepcm targets to be built.
# N.B. We don't link with libCling on Windows, since linking with libCling leads to the following error:
#   MSVCRT.lib(tncleanup.obj) : error LNK2005: "struct __type_info_node __type_info_root_node"
#   (?__type_info_root_node@@3U__type_info_node@@A) already defined in libCling.lib(libCling.dll)
#   [C:\build\workspace\root-pullrequests-build\build\core\base\test\CoreBaseTests.vcxproj]

if(NOT MSVC)
  set(extralibs Cling dl)
endif()

ROOT_ADD_GTEST(CoreBaseTests
  TNamedTests.cxx
  TQObjectTests.cxx
  TExceptionHandlerTests.cxx
  TStringTest.cxx
  TBitsTests.cxx
  LIBRARIES ${extralibs} RIO Core)

ROOT_ADD_GTEST(CoreErrorTests TErrorTests.cxx LIBRARIES Core)
