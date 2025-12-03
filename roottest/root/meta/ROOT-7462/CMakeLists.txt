ROOTTEST_GENERATE_DICTIONARY(instlib
                             instHeader.h
                             instGood.cxx
                             LINKDEF linkdef.h
                             NO_CXXMODULE
                             FIXTURES_SETUP root-meta-7462-instlib-fixtures)

ROOTTEST_GENERATE_DICTIONARY(inst2lib
                             instHeader.h
                             instGood.cxx
                             LINKDEF secondLinkdef.h
                             NO_CXXMODULE NO_ROOTMAP
                             FIXTURES_SETUP root-meta-7462-inst2lib-fixtures)

ROOTTEST_ADD_TEST(WriteFile
                  MACRO execWriteFile.C
                  OUTREF execWriteFile.ref
                  FIXTURES_REQUIRED root-meta-7462-instlib-fixtures
                                    root-meta-7462-inst2lib-fixtures
                  FIXTURES_SETUP root-meta-7462-WriteFile-fixtures)

ROOTTEST_ADD_TEST(Test
                  MACRO execTest.C
                  OUTREF execTest.ref
                  FIXTURES_REQUIRED root-meta-7462-WriteFile-fixtures)

ROOTTEST_ADD_TEST(InvalidDeclRecoveryCI
                  MACRO execInvalidDeclRecoveryCI.C
                  OUTREF execInvalidDeclRecoveryCI.ref)

# This tests intentionally output error message about failed interpreter 
# compilation, so we do not record the log and only check the exit code. 
ROOTTEST_ADD_TEST(InvalidDeclRecovery
                  MACRO runInvalidDeclRecovery.C)
