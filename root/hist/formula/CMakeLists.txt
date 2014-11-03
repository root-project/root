ROOTTEST_ADD_TEST(roottest-root-hist-formula-assertTernary MACRO assertTernary.C OUTREF assertTernary.ref)
ROOTTEST_ADD_TEST(roottest-root-hist-formula-execGetExp MACRO execGetExp.C OUTREF execGetExp.ref)
ROOTTEST_ADD_TEST(roottest-root-hist-formula-runconstargs MACRO runconstargs.C+ OUTREF_CINTSPECIFIC constargs.ref)
ROOTTEST_ADD_TEST(roottest-root-hist-formula-runformio
                  MACRO runformio.C
                  COPY_TO_BUILDDIR result_30gev_sep05.root)
ROOTTEST_ADD_TEST(roottest-root-hist-formula-runrecurse MACRO runrecurse.C OUTREF recurse.ref)
ROOTTEST_ADD_TEST(roottest-root-hist-formula-runstring MACRO runstring.C OUTREF string.ref)
