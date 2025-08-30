ROOTTEST_GENERATE_REFLEX_DICTIONARY(libatlas00
                                    classes00.hh
                                    SELECTION sel00.xml
                                    FIXTURES_SETUP root-meta-autoloading-ROOT-12378-libatlas00-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(libatlas01
                                    classes01.hh
                                    SELECTION sel01.xml
                                    FIXTURES_SETUP root-meta-autoloading-ROOT-12378-libatlas01-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(libatlas02
                                    classes02.hh
                                    SELECTION sel02.xml
                                    FIXTURES_SETUP root-meta-autoloading-ROOT-12378-libatlas02-fixture)

ROOTTEST_ADD_TEST(testtypedef
                  MACRO exec_testtypedef.C
                  FIXTURES_REQUIRED root-meta-autoloading-ROOT-12378-libatlas00-fixture
                                    root-meta-autoloading-ROOT-12378-libatlas01-fixture
                                    root-meta-autoloading-ROOT-12378-libatlas02-fixture)
