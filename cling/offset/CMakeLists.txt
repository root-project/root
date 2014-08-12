ROOTTEST_ADD_TEST(roottest-cling-offset-vbase-interpreted
                  MACRO runvbase.C
                  OUTREF vbase-i.out.ref
                  ERRREF vbase-i.err.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-offset-vbase-compiled
                  MACRO runvbase.C+
                  OUTREF vbase-c.out.ref
                  ERRREF vbase-c.err.ref
                  LABELS roottest regression cling)

