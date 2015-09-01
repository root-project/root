ROOTTEST_ADD_TEST(roottest-cling-template-longlong
                  MACRO runlonglong.C
                  OUTREF longlong.ref
                  DEPENDS longlong.C
                  LABELS roottest regression cling)


ROOTTEST_ADD_TEST(roottest-cling-template-templateSingleton
                  MACRO runtemplateSingleton.C
                  OUTREF templateSingleton.ref
                  DEPENDS Singleton.h
                  LABELS roottest regression cling)

ROOTTEST_GENERATE_DICTIONARY(masterDict       master.h       LINKDEF masterLinkDef.h)
ROOTTEST_GENERATE_DICTIONARY(slaveDict        slave.h        LINKDEF slaveLinkDef.h)
ROOTTEST_GENERATE_DICTIONARY(forwardDict      forward.C      LINKDEF linkdef.h)
ROOTTEST_GENERATE_DICTIONARY(constructorDict  constructor.hh LINKDEF linkdef.h)
ROOTTEST_GENERATE_DICTIONARY(typenameTestDict typenameTest.C LINKDEF linkdef.h)

ROOTTEST_ADD_TEST(roottest-cling-template-runusingtemplate
                  MACRO runusingtemplate.C
                  OUTREF usingtemplate.ref
                  DEPENDS usingtemplate.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-template-runtemplatefriend
                  MACRO runtemplatefriend.C
                  OUTREF templatefriend.ref
                  DEPENDS templatefriend.cxx
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-template-smalltest-interpreted
                  MACRO smalltest.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-template-smalltest-compiled
                  MACRO smalltest.C+
                  LABELS roottest regression cling)

ROOTTEST_ADD_TESTDIRS()
