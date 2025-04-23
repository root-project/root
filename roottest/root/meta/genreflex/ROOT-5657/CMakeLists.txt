
# -Werror makes no sense with this test; remove it:
string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes classes.h SELECTION classes_selection.xml)

ROOTTEST_ADD_TEST(execloadLib
                  MACRO execloadLib.C
                  OUTREF execloadLib.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
