ROOTTEST_ADD_TESTDIRS()

ROOTTEST_ADD_TEST(customRootrcPath
                  MACRO customRootrcPath.C
                  ENVIRONMENT ROOTENV_USER_PATH=${CMAKE_CURRENT_SOURCE_DIR}/customrootc)

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

# In the tests below, root might not have been installed yet, so we can't rely on rootsys and etcdir --> ROOTIGNOREPREFIX
add_test(NAME root_exe_exitStatus COMMAND $<TARGET_FILE:root.exe> "-e invalid" "--no_exist")
set_tests_properties(root_exe_exitStatus PROPERTIES
  WILL_FAIL True
  ENVIRONMENT ROOTIGNOREPREFIX=1)
add_test(NAME root_exe_errorMessage COMMAND $<TARGET_FILE:root.exe> "-e invalid" "--no_exist")
set_tests_properties(root_exe_errorMessage PROPERTIES
             PASS_REGULAR_EXPRESSION "root: unrecognized option '-e invalid'.*--no_exist.*Try 'root --help'"
             ENVIRONMENT ROOTIGNOREPREFIX=1)
