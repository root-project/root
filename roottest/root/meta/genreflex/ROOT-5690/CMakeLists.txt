ROOTTEST_GENERATE_REFLEX_DICTIONARY(FastAllocVector
                                    FastAllocVector.h
                                    SELECTION FastAllocVector_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-5690-fixture)

ROOTTEST_ADD_TEST(execdummy
                  MACRO execdummy.C
                  OUTREF execdummy.ref
                  FIXTURES_REQUIRED root-meta-genreflex-5690-fixture)
