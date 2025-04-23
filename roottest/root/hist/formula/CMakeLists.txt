ROOTTEST_ADD_TEST(assertTernary MACRO assertTernary.C OUTREF assertTernary.ref)
ROOTTEST_ADD_TEST(execGetExp MACRO execGetExp.C OUTREF execGetExp.ref)
ROOTTEST_ADD_TEST(runformio
                  MACRO runformio.C
                  COPY_TO_BUILDDIR result_30gev_sep05.root)
ROOTTEST_ADD_TEST(runrecurse MACRO runrecurse.C OUTREF recurse.ref)
ROOTTEST_ADD_TEST(runstring MACRO runstring.C OUTREF string.ref)
