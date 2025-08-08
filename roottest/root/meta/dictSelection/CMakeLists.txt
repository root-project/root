ROOTTEST_GENERATE_REFLEX_DICTIONARY(AtlasTest
                                    AtlasTest.h
                                    SELECTION AtlasTest_selection.xml
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-meta-dictSelection-AtlasTest-fixture)

ROOTTEST_ADD_TEST(execAtlasTest
                  MACRO execAtlasTest.C
                  OUTREF execAtlasTest.ref
                  FIXTURES_REQUIRED root-meta-dictSelection-AtlasTest-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(AtlasTest2
                                    AtlasTest2.h
                                    SELECTION AtlasTest2_selection.xml
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-meta-dictSelection-AtlasTest2-fixture)

ROOTTEST_ADD_TEST(execAtlasTest2
                  MACRO execAtlasTest2.C
                  OUTREF execAtlasTest2.ref
                  FIXTURES_REQUIRED root-meta-dictSelection-AtlasTest2-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(classesDictSelection
                                    classesDictSelection.h
                                    SELECTION classesDictSelection_selection.xml
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-meta-dictSelection-classesDictSelection-fixture)

ROOTTEST_ADD_TEST(classesDictSelection
                  MACRO execDictSelection.C
                  OUTREF execDictSelection.ref
                  ${WILLFAIL_ON_WIN32}
                  FIXTURES_REQUIRED root-meta-dictSelection-classesDictSelection-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(Dict_ex2
                                    Dict_ex2.h
                                    SELECTION Dict_ex2_selection.xml
                                    NO_ROOTMAP)
