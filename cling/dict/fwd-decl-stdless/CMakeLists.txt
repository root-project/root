# FIXME: Temporary workaround for runtime_cxxmodule
if(NOT ROOT_runtime_cxxmodules_FOUND)
ROOTTEST_GENERATE_DICTIONARY(lessyDict       lessy.h       LINKDEF lessyLinkDef.h)
ROOTTEST_ADD_TEST(execLessyTest
                  MACRO execLessyTest.C
                  DEPENDS lessyDict-build
                  LABELS roottest regression cling)
endif()
