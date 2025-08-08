ROOTTEST_GENERATE_REFLEX_DICTIONARY(Tau
                                    Tau.h
                                    SELECTION Tau_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-5709-fixture)

ROOTTEST_ADD_TEST(execlibLoad
                  MACRO execLibLoad.C
                  OUTREF execLibLoad.ref
                  FIXTURES_REQUIRED root-meta-genreflex-5709-fixture)
