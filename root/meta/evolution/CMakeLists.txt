foreach(_i 1 2 3 4 5 6)
  ROOTTEST_ADD_TEST(roottest-root-meta-evolution-data${_i}
                    MACRO writefile.C
                    MACROARG ${_i}
                    DEPENDS data${_i}.C)
  list(APPEND _data_depends roottest-root-meta-evolution-data${_i})
endforeach()

foreach(_i 0 1 2 3 4 10 11 12 13 14 15 17 18)
  ROOTTEST_ADD_TEST(roottest-root-meta-evolution-checkWarn${_i}
                  MACRO checkWarn.C
                  MACROARG ${_i}
                  DEPENDS ${_data_depends})
endforeach()


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
