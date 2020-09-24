ROOTTEST_ADD_TEST(reloadvector
                  MACRO runreloadvector.C
                  OUTREF reloadvector.ref
                  LABELS roottest regression cling)
                  
                  
ROOTTEST_ADD_TEST(ROOT-7364
                  MACRO assertROOT7364.C
                  LABELS roottest regression cling)
