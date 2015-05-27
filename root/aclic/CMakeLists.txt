ROOTTEST_ADD_TESTDIRS()

# Define the test from parent directory since add_test() seems to ignore it when is defined there
 ROOTTEST_ADD_TEST(roottest-root-aclic-withspace-runthetest
                  MACRO with\ space/runthetest.C
                  OUTREF with\ space/thetest.ref)
