add_definitions(
  -DClingWorkAroundAutoParseTooPrecise
  -DClingWorkAroundAutoParseDeclaration
  -DClingWorkAroundMissingDynamicScope
)

ROOTTEST_COMPILE_MACRO(data1.C)
ROOTTEST_ADD_TEST(roottest-root-meta-evolution-data1
                  MACRO writefile.C
                  MACROARG 1
                  DEPENDS ${COMPILE_MACRO_TEST})

ROOTTEST_COMPILE_MACRO(data2.C)
ROOTTEST_ADD_TEST(roottest-root-meta-evolution-data2
                  MACRO writefile.C
                  MACROARG 2
                  DEPENDS ${COMPILE_MACRO_TEST})

ROOTTEST_COMPILE_MACRO(data3.C)
ROOTTEST_ADD_TEST(roottest-root-meta-evolution-data3
                  MACRO writefile.C
                  MACROARG 3
                  DEPENDS ${COMPILE_MACRO_TEST})

ROOTTEST_COMPILE_MACRO(data4.C)
ROOTTEST_ADD_TEST(roottest-root-meta-evolution-data4
                  MACRO writefile.C
                  MACROARG 4
                  DEPENDS ${COMPILE_MACRO_TEST})

ROOTTEST_COMPILE_MACRO(data5.C)
ROOTTEST_ADD_TEST(roottest-root-meta-evolution-data5
                  MACRO writefile.C
                  MACROARG 5
                  DEPENDS ${COMPILE_MACRO_TEST})

ROOTTEST_COMPILE_MACRO(data6.C)
ROOTTEST_ADD_TEST(roottest-root-meta-evolution-data6
                  MACRO writefile.C
                  MACROARG 6
                  DEPENDS ${COMPILE_MACRO_TEST})

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-checkWarn0
                  MACRO checkWarn.C
                  MACROARG 0)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-checkWarn1
                  MACRO checkWarn.C
                  MACROARG 1)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-checkWarn2
                  MACRO checkWarn.C
                  MACROARG 2)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-checkWarn3
                  MACRO checkWarn.C
                  MACROARG 3)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-checkWarn4
                  MACRO checkWarn.C
                  MACROARG 4)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-checkWarn10
                  MACRO checkWarn.C
                  MACROARG 10)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-checkWarn11
                  MACRO checkWarn.C
                  MACROARG 11)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-checkWarn12
                  MACRO checkWarn.C
                  MACROARG 12)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-checkWarn13
                  MACRO checkWarn.C
                  MACROARG 13)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-checkWarn14
                  MACRO checkWarn.C
                  MACROARG 14)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-checkWarn15
                  MACRO checkWarn.C
                  MACROARG 15)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-checkWarn17
                  MACRO checkWarn.C
                  MACROARG 17)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-checkWarn18
                  MACRO checkWarn.C
                  MACROARG 18)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-execBaseClass_v1
                  MACRO execBaseClass_v1.cxx+
                  OUTREF execBaseClass_v1.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-execBaseClass_v2
                  MACRO execBaseClass_v2.cxx+
                  OUTREF execBaseClass_v2.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-execReadOldCSC
                  MACRO execReadOldCSC.C
                  OUTREF execReadOldCSC.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-execCheckSum
                  MACRO execCheckSum.cxx+
                  OUTREF execCheckSum.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-execMissingCheckSum
                  MACRO execMissingCheckSum.cxx+
                  OUTREF execMissingCheckSum.ref)
