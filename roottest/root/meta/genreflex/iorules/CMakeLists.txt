ROOTTEST_GENERATE_REFLEX_DICTIONARY(DataModelV2genreflex
                                    DataModelV2genreflex.h
                                    SELECTION DataModelV2genreflex_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-iorules-DataModelV2genreflex-fixture)

ROOTTEST_ADD_TEST(DataModelV2genreflex
                  MACRO execSimpleIoRule.C
                  OUTREF execSimpleIoRule.ref
                  FIXTURES_REQUIRED root-meta-genreflex-iorules-DataModelV2genreflex-fixture)
