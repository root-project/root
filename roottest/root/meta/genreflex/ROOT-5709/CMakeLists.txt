ROOTTEST_GENERATE_REFLEX_DICTIONARY(Tau Tau.h SELECTION Tau_selection.xml)

ROOTTEST_ADD_TEST(execlibLoad
                  MACRO execLibLoad.C
                  OUTREF execLibLoad.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
