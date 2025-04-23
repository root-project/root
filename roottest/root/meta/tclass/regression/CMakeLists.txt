ROOTTEST_ADD_TEST(6020
                  MACRO runROOT-6020.C
                  OUTREF ROOT-6020.ref)

ROOTTEST_ADD_TEST(execNormalizationInf
                  MACRO execNormalizationInf.C
                  OUTREF execNormalizationInf.ref)

if(ROOT_pyroot_FOUND)
    ROOTTEST_ADD_TEST(execNormalizationInfPy
                  MACRO execNormalizationInf.py
                  OUTREF execNormalizationInf.py.ref
                  ENVIRONMENT CLING_STANDARD_PCH=none
                  CPPYY_BACKEND_LIBRARY=${CMAKE_BINARY_DIR}/lib/libcppyy_backend.so)
endif()

ROOTTEST_ADD_TEST(execROOT_6038
                  MACRO execROOT_6038.C
                  OUTREF execROOT_6038.ref)

ROOTTEST_ADD_TEST(execROOT_6019
                  MACRO execROOT_6019.C
                  OUTREF execROOT_6019.ref)

if(MSVC)
    # needed on Windows to prevent having to run the test twice
    # the first time being always failing and successful the second time
    ROOTTEST_COMPILE_MACRO(execROOT_6277.cxx)
endif()
ROOTTEST_ADD_TEST(execROOT_6277
                  MACRO execROOT_6277.cxx+
                  OUTREF execROOT_6277.ref)
