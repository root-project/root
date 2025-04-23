ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes classes.h SELECTION classes_selection.xml)

#---Copy from source to binary some of the files
configure_file(example.xml . COPYONLY)

ROOTTEST_ADD_AUTOMACROS(DEPENDS ${GENERATE_REFLEX_TEST}
                        EXCLUDE execionameiotype)

ROOTTEST_ADD_TEST(execionameiotype
                  MACRO execionameiotype.C
                  DEPENDS ${GENERATE_REFLEX_TEST}
                  OUTREF execionameiotype.ref)
