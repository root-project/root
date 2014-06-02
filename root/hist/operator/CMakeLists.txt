ROOTTEST_ADD_TEST(roottest-root-hist-operator-rootlogon MACRO rootlogon.C)
ROOTTEST_ADD_TEST(roottest-root-hist-operator-run MACRO run.C)
ROOTTEST_ADD_TEST(roottest-root-hist-operator-runcopy MACRO runcopy.C+ OUTREF copy.ref)
ROOTTEST_ADD_TEST(roottest-root-hist-operator-runProjections MACRO runProjections.C OUTREF Projections.ref)
