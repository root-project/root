ROOTTEST_ADD_AUTOMACROS()

ROOTTEST_ADD_TEST(ROOT8442
                  MACRO ROOT8442.C
                  ERRREF ROOT8442.errref
                  PASSRC 1
                  ${MIGHT_FAIL}
                  LABELS roottest regression cling)

