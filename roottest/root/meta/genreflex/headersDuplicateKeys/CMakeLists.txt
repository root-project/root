ROOTTEST_GENERATE_REFLEX_DICTIONARY(a1
                                    a1.h
                                    SELECTION a1_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-headersDuplicateKeys-a1-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(a2
                                    a2.h
                                    SELECTION a2_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-headersDuplicateKeys-a2-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(a3
                                    a3.h
                                    SELECTION a3_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-headersDuplicateKeys-a3-fixture)

ROOTTEST_ADD_TEST(headersDuplicateKeys
                  MACRO execduplicateHeaders.C
                  OUTREF execduplicateHeaders.ref
                  FIXTURES_REQUIRED root-meta-genreflex-headersDuplicateKeys-a1-fixture
                                    root-meta-genreflex-headersDuplicateKeys-a2-fixture
                                    root-meta-genreflex-headersDuplicateKeys-a3-fixture)
