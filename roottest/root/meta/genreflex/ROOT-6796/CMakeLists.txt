ROOTTEST_GENERATE_REFLEX_DICTIONARY(attributesFromTypedef
                                    attributesFromTypedef.h
                                    SELECTION attributesFromTypedef_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-6796-fixture)

ROOTTEST_ADD_TEST(attributesFromTypedef
                  MACRO execattributesFromTypedef.C
                  OUTREF execattributesFromTypedef.ref
                  FIXTURES_REQUIRED root-meta-genreflex-6796-fixture)
