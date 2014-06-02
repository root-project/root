ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes classes.h SELECTION classes_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-ROOT-5764-execLibLoad
                  MACRO execLibLoad.C
                  OUTREF execLibLoad.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
