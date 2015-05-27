ROOTTEST_ADD_TEST(roottest-root-meta-evolution-data1
                  MACRO writefile.C
                  MACROARG 1
                  DEPENDS data1.C)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-data2
                  MACRO writefile.C
                  MACROARG 2
                  DEPENDS data2.C)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-data3
                  MACRO writefile.C
                  MACROARG 3
                  DEPENDS data3.C)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-data4
                  MACRO writefile.C
                  MACROARG 4
                  DEPENDS data4.C)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-data5
                  MACRO writefile.C
                  MACROARG 5
                  DEPENDS data5.C)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-data6
                  MACRO writefile.C
                  MACROARG 6
                  DEPENDS data6.C)

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
                  OUTREF execBaseClass_v2.ref
                  DEPENDS roottest-root-meta-evolution-execBaseClass_v1 )

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-execMixedBaseClass_v1
                  MACRO execMixedBaseClass_v1.cxx+
                  OUTREF execMixedBaseClass_v1.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-execMixedBaseClass_v2
                  MACRO execMixedBaseClass_v2.cxx+
                  OUTREF execMixedBaseClass_v2.ref
                  DEPENDS roottest-root-meta-evolution-execMixedBaseClass_v1 )

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-execReadOldCSC
                  COPY_TO_BUILDDIR cscdstarmb.root
                  MACRO execReadOldCSC.C
                  OUTREF execReadOldCSC.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-execCheckSum
                  COPY_TO_BUILDDIR checksum_v53418.root checksum_v5.root checksum_v6.root
                  MACRO execCheckSum.cxx+
                  OUTREF execCheckSum.ref)

ROOTTEST_ADD_TEST(roottest-root-meta-evolution-execMissingCheckSum
                  COPY_TO_BUILDDIR missingCheckSum.root missingCheckSum2.root checksumReflexEnum_v5.root 
                                   checksumSignedChar_v5.root
                  MACRO execMissingCheckSum.cxx+
                  OUTREF execMissingCheckSum.ref)

ROOTTEST_ADD_TESTDIRS()
