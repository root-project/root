ROOTTEST_ADD_TEST(runCPPComments
                  MACRO runCPPComments.C
                  OUTREF CPPComments.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(MissingInclude
                  MACRO MissingInclude.C
                  LABELS roottest regression cling)
