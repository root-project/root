add_definitions(-DClingWorkAroundMissingDynamicScope)

ROOTTEST_COMPILE_MACRO(duplicate.C)
ROOTTEST_ADD_TEST(roottest-cling-templateMembers-runduplicate
                  MACRO runduplicate.C
                  OUTREF duplicate.ref
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(operatorEqual.C)
ROOTTEST_ADD_TEST(roottest-cling-templateMembers-runoperatorEqual
                  MACRO runoperatorEqual.C
                  OUTREF operatorEqual.ref
                  OUTCNVCMD grep -v FILE
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(constructor.C)
ROOTTEST_ADD_TEST(roottest-cling-templateMembers-runConstructor
                  MACRO runConstructor.C
                  OUTREF templateConstructorTest.ref
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

ROOTTEST_GENERATE_DICTIONARY(opdict LINKDEF operator.C linkdef.h)
