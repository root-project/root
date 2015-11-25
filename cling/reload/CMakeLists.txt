#if(NOT ROOT_ARCHITECTURE STREQUAL macosx64)
#  set(_willfail WILLFAIL)
#endif()

ROOTTEST_ADD_TEST(reloadvector
                  MACRO runreloadvector.C
                  OUTREF reloadvector.ref
                  ${_willfail}
                  LABELS roottest regression cling)
