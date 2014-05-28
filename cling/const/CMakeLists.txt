ROOTTEST_ADD_TEST(roottest-cling-const-run1
                  MACRO runConst.C
                  OUTREF good1.output
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-const-run2
                  MACRO runConst.C+
                  OUTREF good2.output
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-const-constCharTest
                  MACRO ../../root/build.C\(\"constissue.h\"\)
                  LABELS roottest regression cling)
