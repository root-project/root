ROOTTEST_ADD_TEST(SetProblem
                  MACRO runSetProblem.C
                  COPY_TO_BUILDDIR rootlogon.C set_problem.h set_problem.cxx
                  OUTREF setLong64Test.ref)
