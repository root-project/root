ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes SELECTION classes_selection.xml classes.h)

ROOTTEST_ADD_TEST(exec5627
                  MACRO exec5627.C
                  OUTREF exec5627.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
