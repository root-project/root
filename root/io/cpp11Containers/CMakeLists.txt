ROOTTEST_ADD_TESTDIRS()

ROOTTEST_GENERATE_REFLEX_DICTIONARY(forwardList forwardList.h  SELECTION forwardList_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-io-cpp11Containers-forwardList
                  MACRO  execForwardList.C
                  OUTREF execForwardList.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(unorderedSet unorderedSet.h  SELECTION unorderedSet_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-io-cpp11Containers-unordered_set
                  MACRO  execUnorderedSet.C
                  OUTREF execUnorderedSet.ref
                  DEPENDS ${GENERATE_REFLEX_TEST} roottest-root-io-cpp11Containers-forwardList)
