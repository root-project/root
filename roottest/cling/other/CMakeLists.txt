ROOTTEST_ADD_TEST(runfileClose
                  MACRO runfileClose.C
                  OUTREF fileClose.ref
                  DEPENDS fileClose.C
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(testCompile.C)

ROOTTEST_ADD_TEST(runsavannah54662
                  MACRO runsavannah54662.C
                  OUTREF savannah54662.ref
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runsnprintfselect
                  MACRO runsnprintfselect.C
                  OUTREF snprintfselect.ref
                  ${WILLFAIL_ON_WIN32}
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(staticConstTest
                  MACRO staticConst.C
                  OUTREF staticConstTest.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runStaticDataMember
                  MACRO runStaticDataMember.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runmissingXFunc
                  MACRO runmissingXFunc.C
                  OUTREF missingXFunc.ref
                  WILLFAIL
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(assertBaseMembers
                  MACRO assertBaseMembers.C
                  LABELS roottest regression cling)
                  
ROOTTEST_ADD_TEST(assertRedirect
                  MACRO assertRedirect.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(execValuePrint
                  MACRO execValuePrint.C
                  OUTREF execValuePrint.ref
                  LABELS roottest regression cling)

if(MSVC)
    set(ERREF_FILE checkMissingSymbolExitCode_win32.eref)
else()
    set(ERREF_FILE checkMissingSymbolExitCode.eref)
endif()
ROOTTEST_ADD_TEST(checkMissingSymbolExitCode
                  MACRO checkMissingSymbolExitCode.C
                  OUTREF checkMissingSymbolExitCode.oref
                  ERRREF ${ERREF_FILE}
                  PASSRC 1
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(assertDirWithParen
                  MACRO assertDirWithParen.C
                  LABELS roottest regression cling)
