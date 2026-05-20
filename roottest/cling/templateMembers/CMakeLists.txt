ROOTTEST_COMPILE_MACRO(duplicate.C
                       FIXTURES_SETUP cling-templateMembers-duplicate-fixture)

ROOTTEST_ADD_TEST(runduplicate
                  MACRO runduplicate.C
                  OUTREF duplicate.ref
                  FIXTURES_REQUIRED cling-templateMembers-duplicate-fixture
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(operatorEqual.C
                       FIXTURES_SETUP cling-templateMembers-operatorEqual-fixture)

ROOTTEST_ADD_TEST(runoperatorEqual
                  MACRO runoperatorEqual.C
                  OUTREF_CINTSPECIFIC operatorEqual.ref
                  OUTCNVCMD grep -v FILE
                  OUTCNV runoperatorEqual_convert.sh
                  FIXTURES_REQUIRED cling-templateMembers-operatorEqual-fixture
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(constructor.C
                       FIXTURES_SETUP cling-templateMembers-constructor-fixture)

ROOTTEST_ADD_TEST(runConstructor
                  MACRO runConstructor.C
                  OUTREF templateConstructorTest.ref
                  FIXTURES_REQUIRED cling-templateMembers-constructor-fixture
                  LABELS roottest regression cling)

ROOTTEST_GENERATE_DICTIONARY(opdict operator.C NO_CXXMODULE LINKDEF linkdef.h)

ROOTTEST_GENERATE_DICTIONARY(libtemplateMembersDict
                             templateMembersClasses.h
                             LINKDEF templateMembersLinkdef.h
                             NO_CXXMODULE
                             FIXTURES_SETUP cling-templateMembers-libtemplateMembersDict-fixture)

ROOTTEST_COMPILE_MACRO(templateMembers.C
                       FIXTURES_REQUIRED cling-templateMembers-libtemplateMembersDict-fixture
                       FIXTURES_SETUP cling-templateMembers-templateMembers-fixture)

ROOTTEST_ADD_TEST(runtemplateMembers-interprter
                  MACRO runtemplateMembers.C
                  MACROARG false
                  OUTCNVCMD grep -v -e "Processing" -e "Mem"
                  PRECMD ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/templateMembersCode_C${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/templateMembersCode_C_ACLiC_dict_rdict.pcm
                  FIXTURES_REQUIRED cling-templateMembers-templateMembers-fixture
                                    cling-templateMembers-libtemplateMembersDict-fixture
                  FIXTURES_SETUP cling-templateMembers-runtemplateMembers-interprter-fixture
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(templateMembersCode.C
                        FIXTURES_REQUIRED cling-templateMembers-runtemplateMembers-interprter-fixture
                                          cling-templateMembers-libtemplateMembersDict-fixture
                        FIXTURES_SETUP cling-templateMembers-templateMembersCode-fixture)

ROOTTEST_ADD_TEST(runtemplateMemberscompiled
                  MACRO runtemplateMembers.C
                  MACROARG true
                  OUTCNVCMD grep -v -e "Processing" -e "Mem"
                  FIXTURES_REQUIRED cling-templateMembers-templateMembers-fixture
                                    cling-templateMembers-libtemplateMembersDict-fixture
                                    cling-templateMembers-templateMembersCode-fixture
                  LABELS roottest regression cling)
