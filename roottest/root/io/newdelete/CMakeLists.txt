ROOTTEST_GENERATE_DICTIONARY(libRoottestIoNewdelete
                             Classes.h
                             LINKDEF LinkDef.h
                             SOURCES Classes.cxx
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-io-newdelete-dict-fixture)

ROOTTEST_ADD_TEST(run
                  MACRO runClasses.C
                  OUTREF good.output
                  FIXTURES_REQUIRED root-io-newdelete-dict-fixture)