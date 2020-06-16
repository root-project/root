ROOTTEST_ADD_AUTOMACROS(DEPENDS namingMatches.cxx
                        EXCLUDE execCheckNaming)

if(MSVC)
    set(EXECCHECKNAMING_OUTREF execCheckNaming_win32.ref)
else()
    set(EXECCHECKNAMING_OUTREF execCheckNaming.ref)
endif()

ROOTTEST_ADD_TEST(execCheckNaming
                  MACRO execCheckNaming.C
                  OUTREF ${EXECCHECKNAMING_OUTREF})
