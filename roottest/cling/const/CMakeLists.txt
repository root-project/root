ROOTTEST_ADD_TEST(run1
                  MACRO runConst.C
                  OUTREF good1.output
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(run2
                  MACRO runConst.C+
                  OUTREF good2.output
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(constCharTest
                  MACRO constissue.C
                  LABELS roottest regression cling)
