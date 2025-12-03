if(NOT MSVC)
  set(warnopt -Wno-delete-non-virtual-dtor)
endif()

ROOTTEST_GENERATE_DICTIONARY(libAbstractClasses
                             PureAbstractInterface.h DataBlockBase.h DataBlock1.h DataBlock2.h
                             SOURCES PureAbstractInterface.cxx DataBlockBase.cxx DataBlock1.cxx DataBlock2.cxx
                             COMPILE_OPTIONS ${warnopt}
                             LINKDEF LinkDef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-io-abstractclass-dict-fixture)

ROOTTEST_ADD_TEST(write_test
                  MACRO write_test.C
                  COPY_TO_BUILDDIR DataBlock1.h DataBlock2.h
                  FIXTURES_REQUIRED root-io-abstractclass-dict-fixture
                  FIXTURES_SETUP root-io-abstractclass-write-fixture)

ROOTTEST_ADD_TEST(read_test
                  MACRO read_test.C
                  OUTREF read_test.ref
                  FIXTURES_REQUIRED root-io-abstractclass-write-fixture)

ROOTTEST_ADD_TEST(read_test_v3_05_07
                  MACRO read_test.C
                  MACROARG "\"data_v3_05_07.root\""
                  OUTREF read_test.ref
                  COPY_TO_BUILDDIR data_v3_05_07.root
                  FIXTURES_REQUIRED root-io-abstractclass-write-fixture)

ROOTTEST_ADD_TEST(read_test_v4_00_02
                  MACRO read_test.C
                  MACROARG "\"data_v4_00_02.root\""
                  OUTREF read_test.ref
                  COPY_TO_BUILDDIR data_v4_00_02.root
                  FIXTURES_REQUIRED root-io-abstractclass-write-fixture)

ROOTTEST_COMPILE_MACRO(abstractInVector.C
                       FIXTURES_SETUP root-io-abstractclass-abstractInVector-fixture)

ROOTTEST_ADD_TEST(load_vector
                  MACRO load_vector.C
                  FIXTURES_REQUIRED root-io-abstractclass-abstractInVector-fixture)
