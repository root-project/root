ROOTTEST_GENERATE_REFLEX_DICTIONARY(cmswrapper
                                    cmswrapper.h
                                    SELECTION sel.xml
                                    FIXTURES_SETUP root-meta-autoloading-ROOT-8432-dic-fixture)

ROOTTEST_ADD_TEST(cmsWrapper
                  MACRO execcmsWrapper.C
                  FIXTURES_REQUIRED root-meta-autoloading-ROOT-8432-dic-fixture)

