ROOTTEST_GENERATE_REFLEX_DICTIONARY(basic_allClasses
                                    basic_allClasses.h
                                    SELECTION basic_allClasses_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-TClass-fixture)

ROOTTEST_ADD_TEST(execbasic
                  MACRO execbasic.C
                  OUTREF execbasic.ref
                  FIXTURES_REQUIRED root-meta-genreflex-TClass-fixture)
