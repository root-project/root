ROOTTEST_ADD_TEST(ROOT-7775
                  MACRO  execROOT-7775.C
                  ROOTEXE_OPTS -e "gInterpreter->AddIncludePath(\"-I${CMAKE_CURRENT_SOURCE_DIR}\")"
                  OUTREF execROOT-7775.ref)

ROOTTEST_ADD_TESTDIRS()
