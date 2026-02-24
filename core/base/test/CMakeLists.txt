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
  TUrlTest.cxx
  TBitsTests.cxx
  TROOTTests.cxx
  LIBRARIES ${extralibs} RIO Core)

ROOT_ADD_GTEST(CoreErrorTests TErrorTests.cxx LIBRARIES Core)

ROOT_ADD_GTEST(CoreSystemTests TSystemTests.cxx LIBRARIES Core)

ROOT_ADD_GTEST(CoreColorTests TColorTests.cxx LIBRARIES Core)
if(CMAKE_SYSTEM_NAME MATCHES "Darwin" AND CMAKE_HOST_SYSTEM_VERSION VERSION_GREATER_EQUAL 25.4)
  set_tests_properties(gtest-core-base-CoreColorTests PROPERTIES DISABLED True)
  file(GENERATE
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/run_until_failure
    CONTENT
"import subprocess
for _ in range(100):
    try:
        subprocess.run(['$<TARGET_FILE:CoreColorTests>'], check=True)
    except subprocess.CalledProcessError as e:
        print('This test is expected to fail with a stack corruption: ', e)
        exit(123)
print('This test was expected to fail with a stack corruption. Check if it got fixed.')
exit(1)"
    TARGET CoreColorTests)

  add_test(NAME CoreColor_RunUntilFailure COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/run_until_failure)
  set_tests_properties(CoreColor_RunUntilFailure PROPERTIES SKIP_RETURN_CODE 123)
endif()

configure_file(Foo.C Foo.C COPYONLY)
ROOT_ADD_GTEST(IncludePathTest IncludePathTest.cxx LIBRARIES Core)
