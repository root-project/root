if(NOT 32BIT AND NOT MSVC)
   ROOTTEST_GENERATE_REFLEX_DICTIONARY(forwardList
                                       forwardList.h
                                       SELECTION forwardList_selection.xml
                                       FIXTURES_SETUP root-io-cpp11Containers-forwardListDict-fixture)

   ROOTTEST_ADD_TEST(forwardList
                     MACRO  execForwardList.C
                     OUTREF execForwardList.ref
                     FIXTURES_REQUIRED root-io-cpp11Containers-forwardListDict-fixture
                     FIXTURES_SETUP root-io-cpp11Containers-forwardList-fixture)

   ROOTTEST_GENERATE_REFLEX_DICTIONARY(unorderedSet
                                       unorderedSet.h
                                       SELECTION unorderedSet_selection.xml
                                       FIXTURES_SETUP root-io-cpp11Containers-unorderedSetDict-fixture)

   ROOTTEST_ADD_TEST(unorderedSet
                     MACRO  execUnorderedSet.C
                     OUTREF execUnorderedSet.ref
                     FIXTURES_REQUIRED root-io-cpp11Containers-unorderedSetDict-fixture
                                       root-io-cpp11Containers-forwardList-fixture)
endif()

ROOTTEST_GENERATE_REFLEX_DICTIONARY(unorderedMap
                                    unorderedMap.h
                                    SELECTION unorderedMap_selection.xml
                                    FIXTURES_SETUP root-io-cpp11Containers-unorderedMapDict-fixture)

ROOTTEST_ADD_TEST(unorderedMap
                  MACRO  execUnorderedMap.C
                  OUTREF execUnorderedMap.ref
                  FIXTURES_REQUIRED root-io-cpp11Containers-unorderedMapDict-fixture
                  FIXTURES_SETUP root-io-cpp11Containers-unorderedMap-fixture
                  LABELS longtest)

ROOTTEST_ADD_TEST(Names
                  COPY_TO_BUILDDIR auxCode.h commonUtils.h
                  MACRO  execcpp11ContainersNames.C
                  OUTREF execcpp11ContainersNames.ref
                  FIXTURES_REQUIRED root-io-cpp11Containers-unorderedMap-fixture)
