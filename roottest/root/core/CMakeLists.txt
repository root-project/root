ROOTTEST_ADD_TESTDIRS()

ROOTTEST_ADD_TEST(lifetime
                  MACRO lifetime.C
                  OUTREF lifetime.ref)

ROOTTEST_ADD_TEST(TEnv
                  MACRO  testTEnv.C
                  OUTREF testTEnv.ref)

ROOTTEST_ADD_TEST(stringView
                  MACRO  stringView.C
                  OUTREF stringView.ref)

ROOTTEST_ADD_TEST(base64
                  MACRO  base64.C
                  OUTREF base64.ref)

ROOTTEST_ADD_TEST(HumanReadable
                  MACRO assertHumanReadable.cxx+
                  OUTREF assertHumanReadable.ref)

ROOTTEST_ADD_TEST(ROOT7244
                  MACRO assertROOT7244.C
                  COPY_TO_BUILDDIR subdir_ROOT7244/TheFile.h)

ROOTTEST_ADD_TEST(ROOT8542
                  MACRO assertROOT8542.C)

ROOTTEST_ADD_TEST(StatusBitsCheck
                  MACRO execStatusBitsCheck.C
                  OUTCNV convert.sh
                  OUTREF execStatusBitsCheck.ref)

add_test(NAME root_exe_exitStatus COMMAND $<TARGET_FILE:root.exe> "-e invalid" "--no_exist")
set_property(TEST root_exe_exitStatus PROPERTY WILL_FAIL True)
add_test(NAME root_exe_errorMessage COMMAND $<TARGET_FILE:root.exe> "-e invalid" "--no_exist")
set_property(TEST root_exe_errorMessage PROPERTY
             PASS_REGULAR_EXPRESSION "root: unrecognized option '-e invalid'.*--no_exist.*Try 'root --help'")
