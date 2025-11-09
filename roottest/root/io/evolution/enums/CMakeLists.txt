ROOTTEST_GENERATE_DICTIONARY(libVectorOfEnums
                             classes.hxx
                             LINKDEF classesLinkdef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-io-evolution-enums-dict-fixture)

# At least on Windows, we need to explicit list Core, otherwise the linker will fail.
ROOTTEST_GENERATE_EXECUTABLE(testVectorOfEnums
                             testVectorOfEnums.cxx
                             LIBRARIES Core RIO Tree libVectorOfEnumslibgen
                             FIXTURES_REQUIRED root-io-evolution-enums-dict-fixture
                             FIXTURES_SETUP root-io-evolution-enums-exec-fixture)

ROOTTEST_ADD_TEST(WriteVectorOfNonDefaultEnums
                  EXEC ./testVectorOfEnums
                  OPTS w
                  LABELS roottest regression io
                  FIXTURES_REQUIRED root-io-evolution-enums-exec-fixture
                  FIXTURES_SETUP root-io-evolution-enums-write-fixture)

ROOTTEST_ADD_TEST(ReadVectorOfNonDefaultEnums
                  EXEC ./testVectorOfEnums
                  OPTS t
                  COPY_TO_BUILDDIR test_v634.root  treetest_v634.root
                  OUTREF testVectorOfEnums.ref
                  LABELS roottest regression io
                  FIXTURES_REQUIRED root-io-evolution-enums-exec-fixture
                                    root-io-evolution-enums-write-fixture)

