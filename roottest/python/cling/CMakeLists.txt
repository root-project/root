if(pyroot)
  # On Windows, code JIT-compiled in the interpreter cannot resolve the
  # CppInternal::DispatchRaw::MakeFunctionCallable_interp dispatch slot: it
  # is a data symbol (a function-pointer variable), which the bindexplib
  # generated export tables do not cover. Exporting writable data across a
  # DLL boundary needs a considered fix (or an upstream CppInterOp change
  # to register the dispatch slots by value instead of by symbol lookup).
  if(NOT MSVC OR win_broken_tests)
    set(api_macro_file runPyAPITestCppyy.C) # Uses new Cppyy API
    ROOTTEST_ADD_TEST(api
                      MACRO ${api_macro_file}
                      WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                      OUTREF PyAPITest.ref)
  endif()

  # TPython::LoadMacro and TPython::Import are broken in the new Cppyy
  # https://bitbucket.org/wlav/cppyy/issues/65
  # For now, we rely on our own implementation of TPython
  if(NOT MSVC OR win_broken_tests)
    ROOTTEST_ADD_TEST(class MACRO
                      runPyClassTest.C
                      WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                      OUTREF PyClassTest.ref)
  endif()

  ROOTTEST_ADD_TEST(cling
                    MACRO PyROOT_clingtests.py
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF PyROOT_clingtests.ref)
endif()
