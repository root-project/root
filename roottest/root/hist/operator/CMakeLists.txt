ROOTTEST_ADD_TEST(run
                  MACRO run.C
                  COPY_TO_BUILDDIR add.C)

ROOTTEST_ADD_TEST(runcopy
                  MACRO runcopy.C+
                  OUTREF copy.ref)

ROOTTEST_ADD_TEST(runProjections
                  MACRO runProjections.C
                  OUTREF Projections.ref)

ROOTTEST_ADD_TEST(runsparse
                  MACRO runsparse.C
                  OUTREF sparse.ref)
