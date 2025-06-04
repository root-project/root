ROOTTEST_GENERATE_DICTIONARY(libDatModelV1_dictcint
                  DataModelV1.h
                  LINKDEF DataModelV1LinkDef.h
                  NO_ROOTMAP NO_CXXMODULE
                  FIXTURES_SETUP libDatModelV1_dictcint_fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(libDatModelV1_dictrflx
                  DataModelV1.h
                  SELECTION DataModelV1_selection.xml
                  LIBNAME libDatModelV1_dictrflx
                  NO_ROOTMAP
                  FIXTURES_SETUP libDatModelV1_dictrflx_fixture)

ROOTTEST_GENERATE_DICTIONARY(libDatModelV2_dictcint
                  DataModelV2.h
                  LINKDEF DataModelV2LinkDef.h
                  NO_ROOTMAP NO_CXXMODULE
                  FIXTURES_SETUP libDatModelV2_dictcint_fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(libDatModelV2_dictrflx
                  DataModelV2.h
                  SELECTION DataModelV2_selection.xml
                  LIBNAME libDatModelV2_dictrflx
                  NO_ROOTMAP
                  FIXTURES_SETUP libDatModelV2_dictrflx_fixture)

ROOTTEST_ADD_TEST(test1
                  MACRO test1.cxx+
                  OUTREF test1.ref
                  LABELS longtest
                  FIXTURES_REQUIRED libDatModelV1_dictcint_fixture
                  FIXTURES_SETUP io-datamodelevolution-01-test1)

ROOTTEST_ADD_TEST(rtest1
                  MACRO test1.cxx+
                  MACROARG "\"r\""
                  OUTREF rtest1.ref
                  LABELS longtest
                  FIXTURES_REQUIRED libDatModelV1_dictrflx_fixture io-datamodelevolution-01-test1
                  FIXTURES_SETUP io-datamodelevolution-01-rtest1)

ROOTTEST_ADD_TEST(test2
                  MACRO test2.cxx+
                  OUTREF test2.ref
                  LABELS longtest
                  FIXTURES_REQUIRED io-datamodelevolution-01-test1
                  FIXTURES_SETUP io-datamodelevolution-01-test2)

ROOTTEST_ADD_TEST(rtest2
                  MACRO test2.cxx+
                  MACROARG "\"r\""
                  OUTREF rtest2.ref
                  LABELS longtest
                  FIXTURES_REQUIRED io-datamodelevolution-01-rtest1 io-datamodelevolution-01-test2)

ROOTTEST_ADD_TEST(test3
                  MACRO test3.cxx+
                  OUTREF test3.ref
                  LABELS longtest
                  FIXTURES_REQUIRED libDatModelV2_dictcint_fixture io-datamodelevolution-01-test1
                  FIXTURES_SETUP io-datamodelevolution-01-test3)

ROOTTEST_ADD_TEST(rtest3
                  MACRO test3.cxx+
                  MACROARG "\"r\""
                  OUTREF rtest3.ref
                  WILLFAIL
                  LABELS longtest
                  FIXTURES_REQUIRED libDatModelV2_dictrflx_fixture io-datamodelevolution-01-rtest1 io-datamodelevolution-01-test3)

