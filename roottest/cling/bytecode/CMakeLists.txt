ROOTTEST_ADD_TEST(runarrayhist
                  MACRO runarrayhist.C
                  OUTREF arrayhist.ref
                  WILLFAIL
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runcomplex
                  MACRO runcomplex.C
                  OUTREF complex.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runhenry
                  COPY_TO_BUILDDIR henry.dat
                  MACRO runhenry.C
                  OUTREF henry.ref
                  LABELS roottest regression cling)
