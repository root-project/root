ROOTTEST_ADD_TEST(roottest-root-hist-operator-run
                  MACRO run.C
                  COPY_TO_BUILDDIR add.C)

ROOTTEST_ADD_TEST(roottest-root-hist-operator-runcopy
                  MACRO runcopy.C+
                  OUTREF copy.ref)

ROOTTEST_ADD_TEST(roottest-root-hist-operator-runProjections
                  MACRO runProjections.C
                  OUTREF Projections.ref)

ROOTTEST_ADD_TEST(roottest-root-hist-operator-runsparse
                  MACRO runsparse.C
                  OUTREF sparse.ref)
