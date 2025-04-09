ROOTTEST_GENERATE_REFLEX_DICTIONARY(AtlasTest AtlasTest.h SELECTION AtlasTest_selection.xml NO_ROOTMAP)

ROOTTEST_ADD_TEST(execAtlasTest
                  MACRO execAtlasTest.C
                  OUTREF execAtlasTest.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(AtlasTest2 AtlasTest2.h SELECTION AtlasTest2_selection.xml NO_ROOTMAP)

ROOTTEST_ADD_TEST(execAtlasTest2
                  MACRO execAtlasTest2.C
                  OUTREF execAtlasTest2.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(classesDictSelection classesDictSelection.h SELECTION classesDictSelection_selection.xml NO_ROOTMAP)

ROOTTEST_ADD_TEST(classesDictSelection
                  MACRO execDictSelection.C
                  OUTREF execDictSelection.ref
                  ${WILLFAIL_ON_WIN32}
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(Dict_ex2 Dict_ex2.h SELECTION Dict_ex2_selection.xml NO_ROOTMAP)
