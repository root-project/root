foreach(_i 1 2 3 4 5 6)
  ROOTTEST_ADD_TEST(data${_i}
                    MACRO writefile.C
                    MACROARG ${_i}
                    DEPENDS data${_i}.C)
  list(APPEND _data_depends data${_i})
endforeach()

foreach(_i 0 1 2 3 4 10 11 12 13 14 15 16 17 18)
  ROOTTEST_ADD_TEST(checkWarn${_i}
                  MACRO checkWarn.C
                  MACROARG ${_i}
                  OUTREF warning${_i}.oref
                  ERRREF warning${_i}.eref
                  OUTCNVCMD sed -e s,offset=[[:space:]][[:space:]]*[0-9]*,offset=n/a,g
                  DEPENDS ${_data_depends})
endforeach()

ROOTTEST_ADD_TEST(execBaseClass_v1
                  MACRO execBaseClass_v1.cxx+
                  OUTREF execBaseClass_v1.ref)

ROOTTEST_ADD_TEST(execBaseClass_v2
                  MACRO execBaseClass_v2.cxx+
                  OUTREF execBaseClass_v2.ref
                  DEPENDS execBaseClass_v1 )
                  
ROOTTEST_ADD_TEST(baseClass_53410_v1
                  COPY_TO_BUILDDIR  baseClass_53410_v1.root
                  MACRO execBaseClass_v2.cxx+
                  MACROARG "\"baseClass_53410_v1.root\""
                  OUTREF baseClass_53410_v1.ref
                  DEPENDS execBaseClass_v2)

ROOTTEST_ADD_TEST(baseClass_53410_v2
                  COPY_TO_BUILDDIR  baseClass_53410_v2.root
                  MACRO execBaseClass_v2.cxx+
                  MACROARG "\"baseClass_53410_v2.root\""
                  OUTREF baseClass_53410_v2.ref
                  DEPENDS execBaseClass_v2)
                  
ROOTTEST_ADD_TEST(baseClass_merge1
                  PRECMD hadd -f  baseClass_merge1.root baseClass_v1.root baseClass_v2.root
                  MACRO execBaseClass_v2.cxx+
                  MACROARG "\"baseClass_merge1.root\""
                  OUTREF baseClass_merge1.ref
                  DEPENDS execBaseClass_v2 )

ROOTTEST_ADD_TEST(baseClass_merge2
                  PRECMD hadd -f  baseClass_merge2.root baseClass_v2.root baseClass_v1.root
                  MACRO execBaseClass_v2.cxx+
                  MACROARG "\"baseClass_merge2.root\""
                  OUTREF baseClass_merge2.ref
                  DEPENDS execBaseClass_v2 )
                                    
ROOTTEST_ADD_TEST(execMixedBaseClass_v1
                  MACRO execMixedBaseClass_v1.cxx+
                  OUTREF execMixedBaseClass_v1.ref)

ROOTTEST_ADD_TEST(execMixedBaseClass_v2
                  MACRO execMixedBaseClass_v2.cxx+
                  OUTREF execMixedBaseClass_v2.ref
                  DEPENDS execMixedBaseClass_v1 )
                  
ROOTTEST_ADD_TEST(mixedBase_53410_v1
                  COPY_TO_BUILDDIR  mixedBase_53410_v1.root
                  MACRO execMixedBaseClass_v2.cxx+
                  MACROARG "\"mixedBase_53410_v1.root\""
                  OUTREF mixedBase_53410_v1.ref
                  DEPENDS execMixedBaseClass_v2)

ROOTTEST_ADD_TEST(mixedBase_53410_v2
                  COPY_TO_BUILDDIR  mixedBase_53410_v2.root
                  MACRO execMixedBaseClass_v2.cxx+
                  MACROARG "\"mixedBase_53410_v2.root\""
                  OUTREF mixedBase_53410_v2.ref
                  DEPENDS execMixedBaseClass_v2)
                  
ROOTTEST_ADD_TEST(mixedBase_merge1
                  PRECMD hadd -f mixedBase_merge1.root mixedBase_v1.root mixedBase_v2.root
                  MACRO execMixedBaseClass_v2.cxx+
                  MACROARG "\"mixedBase_merge1.root\""
                  OUTREF mixedBase_merge1.ref
                  DEPENDS execMixedBaseClass_v2 )

ROOTTEST_ADD_TEST(mixedBase_merge2
                  PRECMD hadd -f mixedBase_merge2.root mixedBase_v2.root mixedBase_v1.root
                  MACRO execMixedBaseClass_v2.cxx+
                  MACROARG "\"mixedBase_merge2.root\""
                  OUTREF mixedBase_merge2.ref
                  DEPENDS execMixedBaseClass_v2 )

ROOTTEST_ADD_TEST(execReadOldCSC
                  COPY_TO_BUILDDIR cscdstarmb.root
                  MACRO execReadOldCSC.C
                  OUTREF execReadOldCSC.ref)

ROOTTEST_ADD_TEST(execCheckSum
                  COPY_TO_BUILDDIR checksum_v53418.root checksum_v5.root checksum_v6.root
                  MACRO execCheckSum.cxx+
                  OUTREF execCheckSum.ref)

ROOTTEST_ADD_TEST(execMissingCheckSum
                  COPY_TO_BUILDDIR missingCheckSum.root missingCheckSum2.root checksumReflexEnum_v5.root 
                                   checksumSignedChar_v5.root
                  MACRO execMissingCheckSum.cxx+
                  OUTREF execMissingCheckSum.ref)

ROOTTEST_ADD_TEST(runforeign
                  MACRO runforeign.C
                  OUTREF  foreign.ref
                  DEPENDS data1 data2 data3 data4)
ROOTTEST_ADD_TESTDIRS()
