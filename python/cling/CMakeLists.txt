if(ROOT_pyroot_FOUND)
  if (ROOT_pyroot_legacy_FOUND)
    set(api_macro_file runPyAPITest.C)
  else()
    set(api_macro_file runPyAPITestCppyy.C) # Uses new Cppyy API
  endif()
  ROOTTEST_ADD_TEST(api
                    MACRO ${api_macro_file}
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyAPITest.ref
                    ENVIRONMENT CLING_STANDARD_PCH=none
                    CPPYY_BACKEND_LIBRARY=${CMAKE_BINARY_DIR}/lib/libcppyy_backend${PYTHON_UNDER_VERSION_STRING_Development_Main}.so)

  # TPython::LoadMacro and TPython::Import are broken in the new Cppyy
  # https://bitbucket.org/wlav/cppyy/issues/65
  # For now, we rely on our own implementation of TPython
  if(NOT MSVC OR "${CMAKE_GENERATOR_PLATFORM}" MATCHES "Win32" OR win_broken_tests)
    ROOTTEST_ADD_TEST(class MACRO
                      runPyClassTest.C
                      WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                      OUTREF PyClassTest.ref
                      ENVIRONMENT CLING_STANDARD_PCH=none
                      CPPYY_BACKEND_LIBRARY=${CMAKE_BINARY_DIR}/lib/libcppyy_backend${PYTHON_UNDER_VERSION_STRING_Development_Main}.so)
  endif()

  ROOTTEST_ADD_TEST(cling
                    MACRO PyROOT_clingtests.py
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyROOT_clingtests.ref)
endif()
