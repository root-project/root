# Temporarly add workaround
add_definitions(-DClingWorkAroundAutoParseRecurse) #see https://sft.its.cern.ch/jira/browse/ROOT-6319

ROOTTEST_GENERATE_REFLEX_DICTIONARY(AtlasTest AtlasTest.h SELECTION AtlasTest_selection.xml NO_ROOTMAP)
set(depends ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(AtlasTest2 AtlasTest2.h SELECTION AtlasTest2_selection.xml NO_ROOTMAP)
set(depends ${depends} ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(classesDictSelection classesDictSelection.h SELECTION classesDictSelection_selection.xml NO_ROOTMAP)
set(depends ${depends} ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(Dict_ex2 Dict_ex2.h SELECTION Dict_ex2_selection.xml NO_ROOTMAP)
set(depends ${depends} ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_AUTOMACROS(DEPENDS ${depends})
