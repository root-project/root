ROOTTEST_ADD_TEST(roottest-cling-offset-vbase-interpreted
                  MACRO runvbase.C
                  OUTREF vbase-i.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-offset-vbase-compiled
                  MACRO runvbase.C+
                  OUTREF vbase-c.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-offset-rootlogon
                  MACRO rootlogon.C
                  LABELS roottest regression cling)
