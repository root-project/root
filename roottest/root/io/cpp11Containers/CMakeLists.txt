if(NOT 32BIT AND NOT MSVC)
   ROOTTEST_GENERATE_REFLEX_DICTIONARY(forwardList forwardList.h  SELECTION forwardList_selection.xml)

   ROOTTEST_ADD_TEST(forwardList
                     MACRO  execForwardList.C
                     OUTREF execForwardList.ref
                     DEPENDS ${GENERATE_REFLEX_TEST})

   ROOTTEST_GENERATE_REFLEX_DICTIONARY(unorderedSet unorderedSet.h  SELECTION unorderedSet_selection.xml)

   ROOTTEST_ADD_TEST(unorderedSet
                     MACRO  execUnorderedSet.C
                     OUTREF execUnorderedSet.ref
                     DEPENDS ${GENERATE_REFLEX_TEST} forwardList)
endif()

ROOTTEST_GENERATE_REFLEX_DICTIONARY(unorderedMap unorderedMap.h  SELECTION unorderedMap_selection.xml)

ROOTTEST_ADD_TEST(unorderedMap
                  MACRO  execUnorderedMap.C
                  OUTREF execUnorderedMap.ref
                  DEPENDS ${GENERATE_REFLEX_TEST} unorderedSet
                  LABELS longtest)

ROOTTEST_ADD_TEST(Names
                  COPY_TO_BUILDDIR auxCode.h commonUtils.h
                  MACRO  execcpp11ContainersNames.C
                  OUTREF execcpp11ContainersNames.ref
                  DEPENDS ${GENERATE_REFLEX_TEST} unorderedMap)
