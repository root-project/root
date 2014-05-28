ROOTTEST_ADD_TEST(roottest-cling-return-constRetrunTest
                  MACRO run.C
                  WILLFAIL
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-return-barna
                  MACRO barna.C
                  OUTREF barna.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-return-exitcode
                  MACRO exitcode.C
                  PASSRC 12
                  LABELS roottest regression cling)
