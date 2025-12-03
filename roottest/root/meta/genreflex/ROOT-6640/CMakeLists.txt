ROOTTEST_GENERATE_REFLEX_DICTIONARY(ROOT6440
                                    ROOT6440.h
                                    SELECTION ROOT6440_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-6440-fixture)

ROOTTEST_ADD_TEST(ROOT6440
                  MACRO runROOT6440.C
                  FIXTURES_REQUIRED root-meta-genreflex-6440-fixture)
