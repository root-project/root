ROOTTEST_GENERATE_DICTIONARY(nextDict       next.h       LINKDEF nextLinkDef.h)
set(depends ${GENERATE_DICTIONARY_TEST})

ROOTTEST_ADD_TEST(execNextTest
                  MACRO execNextTest.C
                  OUTREF NextTest.ref
                  DEPENDS $(depends)
                  LABELS roottest regression cling)


