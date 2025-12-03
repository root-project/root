ROOTTEST_COMPILE_MACRO(na.cxx
                       FIXTURES_SETUP root-io-multipleInherit-na-fixture)

ROOTTEST_ADD_TEST(write
                  MACRO wr.C
                  ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L na.cxx+\")"
                  OUTREF write.ref
                  FIXTURES_REQUIRED root-io-multipleInherit-na-fixture
                  FIXTURES_SETUP root-io-multipleInherit-write)

ROOTTEST_ADD_TEST(run
                  MACRO re.C
                  OUTREF run.ref
                  FIXTURES_REQUIRED root-io-multipleInherit-write)
