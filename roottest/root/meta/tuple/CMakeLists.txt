ROOTTEST_GENERATE_REFLEX_DICTIONARY(libConfigType
                                    ConfigType.hpp
                                    SELECTION ConfigType_selection.xml
                                    LIBNAME libConfigType
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-meta-tuple-dict-fixture)

ROOTTEST_ADD_TEST(configtype_unload
                  MACRO exec_configtype_unload.C
                  FIXTURES_REQUIRED root-meta-tuple-dict-fixture)

