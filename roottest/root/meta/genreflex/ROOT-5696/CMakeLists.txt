# -Werror makes no sense with this testing a warning; remove it:
string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(warnMe warnMe.h SELECTION warnMe_selection.xml)

ROOTTEST_ADD_TEST(exec5596
                  MACRO exec5596.C
                  OUTREF exec5596.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
