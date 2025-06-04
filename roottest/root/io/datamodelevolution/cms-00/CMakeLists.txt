ROOTTEST_ADD_TEST(WriteRename
                  MACRO execWriteRename.cxx+
                  OUTREF execWriteRename.ref
                  LABELS longtest
                  FIXTURES_SETUP io-datamodelevolution-cms-00-WriteRename)

ROOTTEST_ADD_TEST(ReadRename1
                  MACRO execReadRename1.cxx+
                  OUTREF execReadRename1.ref
                  LABELS longtest
                  FIXTURES_REQUIRED io-datamodelevolution-cms-00-WriteRename)

ROOTTEST_ADD_TEST(ReadRename2
                  MACRO execReadRename2.cxx+
                  OUTREF execReadRename2.ref
                  LABELS longtest
                  FIXTURES_REQUIRED io-datamodelevolution-cms-00-WriteRename)

ROOTTEST_GENERATE_DICTIONARY(libDataModelV1_dictcint
                  DataModelV1.h
                  LINKDEF DataModelV1LinkDef.h
                  NO_ROOTMAP NO_CXXMODULE
                  FIXTURES_SETUP libDataModelV1_dictcint_fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(libDataModelV1_dictrflx
                  DataModelV1.h
                  SELECTION DataModelV1_selection.xml
                  LIBNAME libDataModelV1_dictrflx
                  NO_ROOTMAP
                  FIXTURES_SETUP libDataModelV1_dictrflx_fixture)

ROOTTEST_GENERATE_DICTIONARY(libDataModelV2_dictcint
                  DataModelV2.h
                  LINKDEF DataModelV2LinkDef.h
                  NO_ROOTMAP NO_CXXMODULE
                  FIXTURES_SETUP libDataModelV2_dictcint_fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(libDataModelV2_dictrflx
                  DataModelV2.h
                  SELECTION DataModelV2_selection.xml
                  LIBNAME libDataModelV2_dictrflx
                  NO_ROOTMAP
                  FIXTURES_SETUP libDataModelV2_dictrflx_fixture)

ROOTTEST_ADD_TEST(test1
                  MACRO test1.cxx+
                  OUTREF test1.ref
                  LABELS longtest
                  FIXTURES_REQUIRED libDataModelV1_dictcint_fixture
                  FIXTURES_SETUP io-datamodelevolution-cms-00-test1)

ROOTTEST_ADD_TEST(reflex_test1
                  MACRO test1.cxx+
                  MACROARG "\"r\""
                  OUTREF reflex_test1.ref
                  LABELS longtest
                  FIXTURES_REQUIRED libDataModelV1_dictrflx_fixture io-datamodelevolution-cms-00-test1
                  FIXTURES_SETUP io-datamodelevolution-cms-00-reflex-test1)

ROOTTEST_ADD_TEST(read_test1
                  MACRO read_test1.cxx+
                  OUTREF read_test1.ref
                  LABELS longtest
                  FIXTURES_REQUIRED io-datamodelevolution-cms-00-test1
                  FIXTURES_SETUP io-datamodelevolution-cms-00-read_test1)

ROOTTEST_ADD_TEST(read_reflex_test1
                  MACRO read_test1.cxx+
                  MACROARG "\"r\""
                  OUTREF read_reflex_test1.ref
                  LABELS longtest
                  FIXTURES_REQUIRED io-datamodelevolution-cms-00-reflex-test1 io-datamodelevolution-cms-00-read_test1)

if(NOT ClingWorkAroundNoPrivateClassIO)
   ROOTTEST_ADD_TEST(test2
                     MACRO test2.cxx+
                     OUTREF test2.ref
                     WILLFAIL
                     LABELS longtest
                     FIXTURES_SETUP io-datamodelevolution-cms-00-test2
                     FIXTURES_REQUIRED libDataModelV2_dictcint_fixture io-datamodelevolution-cms-00-test1)

   ROOTTEST_ADD_TEST(reflex_test2
                     MACRO test2.cxx+
                     MACROARG "\"r\""
                     OUTREF reflex_test2.ref
                     WILLFAIL
                     LABELS longtest
                     FIXTURES_REQUIRED libDataModelV2_dictrflx_fixture io-datamodelevolution-cms-00-reflex-test1 io-datamodelevolution-cms-00-test2)

   ROOTTEST_ADD_TEST(test3
                     MACRO test3.cxx+
                     OUTREF test3.ref
                     WILLFAIL
                     LABELS longtest
                     FIXTURES_SETUP io-datamodelevolution-cms-00-test3
                     FIXTURES_REQUIRED libDataModelV2_dictcint_fixture io-datamodelevolution-cms-00-test1)

   ROOTTEST_ADD_TEST(reflex_test3
                     MACRO test3.cxx+
                     MACROARG "\"r\""
                     OUTREF reflex_test3.ref
                     WILLFAIL
                     LABELS longtest
                     FIXTURES_REQUIRED libDataModelV2_dictrflx_fixture io-datamodelevolution-cms-00-reflex-test1 io-datamodelevolution-cms-00-test3)
endif()