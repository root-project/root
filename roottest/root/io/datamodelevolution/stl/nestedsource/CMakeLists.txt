# Reading a split TTree branch holding a vector<T>, where an I/O rule for T
# sources a member nested inside a struct.
# See https://github.com/root-project/root/issues/19773

ROOTTEST_COMPILE_MACRO(execWriteNestedSource.cxx
                  FIXTURES_SETUP root-io-datamodelevolution-stl-nestedsource-WriteNestedSource-compile-fixture)

ROOTTEST_COMPILE_MACRO(execReadNestedSource.cxx
                  FIXTURES_SETUP root-io-datamodelevolution-stl-nestedsource-ReadNestedSource-compile-fixture)

ROOTTEST_ADD_TEST(WriteNestedSource
                  MACRO execWriteNestedSource.cxx+
                  OUTREF execWriteNestedSource.ref
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-datamodelevolution-stl-nestedsource-WriteNestedSource-compile-fixture
                  FIXTURES_SETUP root-io-datamodelevolution-stl-nestedsource-WriteNestedSource-fixture)

ROOTTEST_ADD_TEST(ReadNestedSource
                  MACRO execReadNestedSource.cxx+
                  OUTREF execReadNestedSource.ref
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-datamodelevolution-stl-nestedsource-ReadNestedSource-compile-fixture
                                    root-io-datamodelevolution-stl-nestedsource-WriteNestedSource-fixture)
