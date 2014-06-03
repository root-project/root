add_definitions(-DClingWorkAroundMissingDynamicScope)

ROOTTEST_ADD_TEST(roottest-cling-templateMembers-runduplicate
                  MACRO runduplicate.C
                  OUTREF duplicate.ref
                  DEPENDS duplicate.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-templateMembers-runoperatorEqual
                  MACRO runoperatorEqual.C
                  OUTREF operatorEqual.ref
                  OUTCNVCMD grep -v FILE
                  DEPENDS operatorEqual.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-templateMembers-runConstructor
                  MACRO runConstructor.C
                  OUTREF templateConstructorTest.ref
                  DEPENDS constructor.C
                  LABELS roottest regression cling)

ROOTTEST_GENERATE_DICTIONARY(opdict LINKDEF operator.C linkdef.h)
