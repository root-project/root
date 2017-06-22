ROOTTEST_ADD_TEST(nullderef-macro
                  MACRO nullderef.C
                  PASSRC 1
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(nullderef-e
                  EXEC root
                  OPTS -l -b -q -e 'int*p=nullptr;*p'
                  PASSRC 1
                  LABELS roottest regression cling)
