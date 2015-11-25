ROOTTEST_ADD_TEST(6020
                  MACRO runROOT-6020.C
                  OUTREF ROOT-6020.ref)

ROOTTEST_ADD_TEST(execNormalizationInf
                  MACRO execNormalizationInf.C
                  OUTREF execNormalizationInf.ref)

ROOTTEST_ADD_TEST(execNormalizationInfPy
                  MACRO execNormalizationInf.py
                  OUTREF execNormalizationInf.py.ref)

ROOTTEST_ADD_TEST(execROOT_6038
                  MACRO execROOT_6038.C
                  OUTREF execROOT_6038.ref)

ROOTTEST_ADD_TEST(execROOT_6019
                  MACRO execROOT_6019.C
                  OUTREF execROOT_6019.ref)

ROOTTEST_ADD_TEST(execROOT_6277
                  MACRO execROOT_6277.cxx+
                  OUTREF execROOT_6277.ref)
