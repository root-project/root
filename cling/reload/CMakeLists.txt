ROOTTEST_ADD_TEST(roottest-cling-reload-reloadvector
                  MACRO runreloadvector.C
                  OUTREF reloadvector.ref
                  WILLFAIL
                  LABELS roottest regression cling)
