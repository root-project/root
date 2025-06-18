ROOTTEST_COMPILE_MACRO(longlong.C
                       FIXTURES_SETUP cling-template-longlong-fixture)

ROOTTEST_ADD_TEST(longlong
                  MACRO runlonglong.C
                  OUTREF longlong.ref
                  FIXTURES_REQUIRED cling-template-longlong-fixture
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(Singleton.h
                       FIXTURES_SETUP cling-template-Singleton-fixture)

ROOTTEST_ADD_TEST(templateSingleton
                  MACRO runtemplateSingleton.C
                  OUTREF templateSingleton.ref
                  FIXTURES_REQUIRED cling-template-Singleton-fixture
                  LABELS roottest regression cling)

if(NOT ROOT_runtime_cxxmodules_FOUND)
  ROOTTEST_GENERATE_DICTIONARY(slaveDict        slave.h        LINKDEF slaveLinkDef.h)
endif()
ROOTTEST_GENERATE_DICTIONARY(masterDict       master.h       LINKDEF masterLinkDef.h)
ROOTTEST_GENERATE_DICTIONARY(forwardDict      forward.C      LINKDEF linkdef.h)
ROOTTEST_GENERATE_DICTIONARY(constructorDict  constructor.hh LINKDEF linkdef.h)
ROOTTEST_GENERATE_DICTIONARY(typenameTestDict typenameTest.C LINKDEF linkdef.h
                             FIXTURES_SETUP cling-template-typenameTestDict-fixture)

ROOTTEST_COMPILE_MACRO(usingtemplate.C
                       FIXTURES_SETUP cling-template-usingtemplate-fixture)

ROOTTEST_ADD_TEST(runusingtemplate
                  MACRO runusingtemplate.C
                  OUTREF usingtemplate.ref
                  FIXTURES_REQUIRED cling-template-usingtemplate-fixture
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(templatefriend.cxx
                       FIXTURES_SETUP cling-template-templatefriend-fixture)

ROOTTEST_ADD_TEST(runtemplatefriend
                  MACRO runtemplatefriend.C
                  OUTREF templatefriend.ref
                  FIXTURES_REQUIRED cling-template-templatefriend-fixture cling-template-typenameTestDict-fixture
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(interpreted
                  MACRO smalltest.C
                  PRECMD ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/smalltest_C${CMAKE_SHARED_LIBRARY_SUFFIX} ${CMAKE_CURRENT_BINARY_DIR}/smalltest_C_ACLiC_dict_rdict.pcm
                  FIXTURES_SETUP cling-template-smalltest-interpreted-fixture
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(compiled
                  MACRO smalltest.C+
                  FIXTURES_REQUIRED cling-template-smalltest-interpreted-fixture
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(defaultTemplateParam
                  MACRO execdefaultTemplateParam.cxx+
                  OUTREF execdefaultTemplateParam.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TESTDIRS()
