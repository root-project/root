# Add defines to fix tests.
add_definitions(
  -DClingWorkAroundMissingDynamicScope
  -DClingWorkAroundUnloadingIOSTREAM
)

ROOTTEST_COMPILE_MACRO(longlong.C)
ROOTTEST_ADD_TEST(roottest-cling-template-longlong
                  MACRO runlonglong.C
                  OUTREF longlong.ref
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)


ROOTTEST_COMPILE_MACRO(Singleton.h)
ROOTTEST_ADD_TEST(roottest-cling-template-templateSingleton
                  MACRO runtemplateSingleton.C
                  OUTREF templateSingleton.ref
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

ROOTTEST_GENERATE_DICTIONARY(masterDict       LINKDEF master.h masterLinkDef.h)
ROOTTEST_GENERATE_DICTIONARY(slaveDict        LINKDEF slave.h slaveLinkDef.h)
ROOTTEST_GENERATE_DICTIONARY(forwardDict      LINKDEF forward.C linkdef.h)
ROOTTEST_GENERATE_DICTIONARY(constructorDict  LINKDEF constructor.hh linkdef.h)
ROOTTEST_GENERATE_DICTIONARY(typenameTestDict LINKDEF typenameTest.C linkdef.h)

ROOTTEST_COMPILE_MACRO(usingtemplate.C)

ROOTTEST_ADD_TEST(roottest-cling-template-runusingtemplate
                  MACRO runusingtemplate.C
                  OUTREF usingtemplate.ref
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(templatefriend.cxx)
ROOTTEST_ADD_TEST(roottest-cling-template-runtemplatefriend
                  MACRO runtemplatefriend.C
                  OUTREF templatefriend.ref
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-template-smalltest-interpreted
                  MACRO smalltest.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-template-smalltest-compiled
                  MACRO smalltest.C+
                  LABELS roottest regression cling)
