ROOTTEST_ADD_TEST(roottest-cling-templateMembers-runduplicate
                  MACRO runduplicate.C
                  OUTREF duplicate.ref
                  DEPENDS duplicate.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-templateMembers-runoperatorEqual
                  MACRO runoperatorEqual.C
                  OUTREF_CINTSPECIFIC operatorEqual.ref
                  OUTCNVCMD grep -v FILE
                  OUTCNV runoperatorEqual_convert.sh
                  DEPENDS operatorEqual.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-templateMembers-runConstructor
                  MACRO runConstructor.C
                  OUTREF templateConstructorTest.ref
                  DEPENDS constructor.C
                  LABELS roottest regression cling)

ROOTTEST_GENERATE_DICTIONARY(opdict operator.C LINKDEF linkdef.h)
