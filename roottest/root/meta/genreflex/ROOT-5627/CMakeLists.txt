ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes
                                    classes.h
                                    SELECTION classes_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-5627-fixture)

ROOTTEST_ADD_TEST(exec5627
                  MACRO exec5627.C
                  OUTREF exec5627.ref
                  FIXTURES_REQUIRED root-meta-genreflex-5627-fixture)
