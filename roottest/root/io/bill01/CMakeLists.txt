ROOTTEST_GENERATE_DICTIONARY(libBill01MyObject
                             MyObject.h
                             LINKDEF LinkDef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-io-bill01-dict-fixture)

ROOTTEST_ADD_TEST(checksize
                  MACRO run.C
                  ROOTEXE_OPTS -e "gInterpreter->AddIncludePath(\"-I${CMAKE_CURRENT_SOURCE_DIR}\")"
                               -e "(void)gSystem->Load(\"libBill01MyObject\")"
                  OUTREF checksize.ref
                  FIXTURES_REQUIRED root-io-bill01-dict-fixture)
