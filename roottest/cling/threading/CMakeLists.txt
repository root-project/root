ROOTTEST_ADD_TEST(clinglock
                  MACRO clinglock.C
                  OPTS -t
                  LABELS roottest cling)

ROOTTEST_ADD_TEST(clinglock_staticinit
                  MACRO clinglock_staticinit.C
                  OPTS -t
                  LABELS roottest cling)
