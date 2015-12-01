#if(NOT ROOT_ARCHITECTURE STREQUAL macosx64)
#  set(_willfail WILLFAIL)
#endif()

ROOTTEST_ADD_TEST(reloadvector
                  MACRO runreloadvector.C
                  OUTREF reloadvector.ref
                  ${_willfail}
                  LABELS roottest regression cling)
                  
                  
ROOTTEST_ADD_TEST(ROOT-7364
                  MACRO assertROOT7364.C
                  LABELS roottest regression cling)
