ROOTTEST_GENERATE_DICTIONARY(libDatModelV1_dictcint
                  DataModelV1.h
                  LINKDEF DataModelV1LinkDef.h
                  NO_ROOTMAP NO_CXXMODULE
                  FIXTURES_SETUP root-io-datamodelevolution-01-libDatModelV1_dictcint-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(libDatModelV1_dictrflx
                  DataModelV1.h
                  SELECTION DataModelV1_selection.xml
                  LIBNAME libDatModelV1_dictrflx
                  NO_ROOTMAP
                  FIXTURES_SETUP root-io-datamodelevolution-01-libDatModelV1_dictrflx-fixture)

ROOTTEST_GENERATE_DICTIONARY(libDatModelV2_dictcint
                  DataModelV2.h
                  LINKDEF DataModelV2LinkDef.h
                  NO_ROOTMAP NO_CXXMODULE
                  FIXTURES_SETUP root-io-datamodelevolution-01-libDatModelV2_dictcint-fixture)


ROOTTEST_COMPILE_MACRO(test1.cxx
                       FIXTURES_SETUP root-io-datamodelevolution-01-test1-compile-fixture)

ROOTTEST_COMPILE_MACRO(test2.cxx
                       FIXTURES_SETUP root-io-datamodelevolution-01-test2-compile-fixture)

ROOTTEST_COMPILE_MACRO(test3.cxx
                       FIXTURES_SETUP root-io-datamodelevolution-01-test3-compile-fixture)


ROOTTEST_ADD_TEST(test1
                  MACRO test1.cxx+
                  OUTREF test1.ref
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-datamodelevolution-01-libDatModelV1_dictcint-fixture
                                    root-io-datamodelevolution-01-test1-compile-fixture
                  FIXTURES_SETUP root-io-datamodelevolution-01-test1-fixture)

ROOTTEST_ADD_TEST(rtest1
                  MACRO test1.cxx+
                  MACROARG "\"r\""
                  OUTREF rtest1.ref
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-datamodelevolution-01-libDatModelV1_dictrflx-fixture
                                    root-io-datamodelevolution-01-test1-compile-fixture
                  FIXTURES_SETUP root-io-datamodelevolution-01-rtest1-fixture)

ROOTTEST_ADD_TEST(test2
                  MACRO test2.cxx+
                  OUTREF test2.ref
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-datamodelevolution-01-test1-fixture
                                    root-io-datamodelevolution-01-test2-compile-fixture)

ROOTTEST_ADD_TEST(rtest2
                  MACRO test2.cxx+
                  MACROARG "\"r\""
                  OUTREF rtest2.ref
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-datamodelevolution-01-rtest1-fixture
                                    root-io-datamodelevolution-01-test2-compile-fixture)

ROOTTEST_ADD_TEST(test3
                  MACRO test3.cxx+
                  OUTREF test3.ref
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-datamodelevolution-01-libDatModelV2_dictcint-fixture
                                    root-io-datamodelevolution-01-test1-fixture
                                    root-io-datamodelevolution-01-test3-compile-fixture)

# problem with XML file parsing on windows
if(NOT MSVC OR win_broken_tests)
  ROOTTEST_GENERATE_REFLEX_DICTIONARY(libDatModelV2_dictrflx
                    DataModelV2.h
                    SELECTION DataModelV2_selection.xml
                    LIBNAME libDatModelV2_dictrflx
                    NO_ROOTMAP
                    FIXTURES_SETUP root-io-datamodelevolution-01-libDatModelV2_dictrflx-fixture)

  ROOTTEST_ADD_TEST(rtest3
                    MACRO test3.cxx+
                    MACROARG "\"r\""
                    OUTREF rtest3.ref
                    LABELS longtest
                    FIXTURES_REQUIRED root-io-datamodelevolution-01-libDatModelV2_dictrflx-fixture
                                      root-io-datamodelevolution-01-rtest1-fixture
                                      root-io-datamodelevolution-01-test3-compile-fixture)
endif()
