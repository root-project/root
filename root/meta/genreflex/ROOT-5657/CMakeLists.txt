ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes classes.h SELECTION classes_selection.xml)

ROOTTEST_ADD_TEST(execloadLib
                  MACRO execloadLib.C
                  OUTREF execloadLib.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
