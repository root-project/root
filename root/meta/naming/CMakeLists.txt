if(NOT MSVC OR llvm13_broken_tests)
  ROOTTEST_ADD_AUTOMACROS(DEPENDS namingMatches.cxx
                          EXCLUDE execCheckNaming)

  ROOTTEST_ADD_TEST(execCheckNaming
                    MACRO execCheckNaming.C
                    OUTREF execCheckNaming${ref_suffix})
endif()
