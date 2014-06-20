ROOTTEST_ADD_TEST(roottest-cling-bytecode-runarrayhist
                  MACRO runarrayhist.C
                  OUTREF arrayhist.ref
                  WILLFAIL
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-bytecode-runcomplex
                  MACRO runcomplex.C
                  OUTREF complex.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-bytecode-runhenry
                  COPY_TO_BUILDDIR henry.dat
                  MACRO runhenry.C
                  OUTREF henry.ref
                  LABELS roottest regression cling)
