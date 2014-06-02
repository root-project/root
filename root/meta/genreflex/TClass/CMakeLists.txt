ROOTTEST_GENERATE_REFLEX_DICTIONARY(basic_allClasses basic_allClasses.h SELECTION basic_allClasses_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-TClass-execbasic
                  MACRO execbasic.C
                  OUTREF execbasic.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
