ROOTTEST_GENERATE_DICTIONARY(myclbad bad.h mycl.C LINKDEF LinkDef.h NO_ROOTMAP)
set(depends ${GENERATE_DICTIONARY_TEST})

ROOTTEST_GENERATE_DICTIONARY(myclgood mycl.C LINKDEF LinkDef.h NO_ROOTMAP)
set(depends ${depends} ${GENERATE_DICTIONARY_TEST})

ROOTTEST_ADD_TEST(scopeTest
                  MACRO run.C
                  OUTREF output.ref-cling
                  OUTCNV scopeTest_conversion.sh
                  DEPENDS ${depends}
                  LABELS roottest regression cling
                  COPY_TO_BUILDDIR helper.h)

ROOTTEST_ADD_TEST(scopeTest2
                  MACRO rungood.C
                  OUTREF output.ref-cling
                  OUTCNV scopeTest_conversion.sh
                  DEPENDS ${depends}
                  LABELS roottest regression cling
                  COPY_TO_BUILDDIR helper.h)
