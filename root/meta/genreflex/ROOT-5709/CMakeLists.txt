ROOTTEST_GENERATE_REFLEX_DICTIONARY(Tau Tau.h SELECTION Tau_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-ROOT-5709-execlibLoad
                  MACRO execLibLoad.C
                  OUTREF execLibLoad.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
