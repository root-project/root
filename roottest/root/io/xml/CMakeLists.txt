
if (xml)
  ROOTTEST_ADD_TEST(SaxParserSimple
                    MACRO execSaxParserSimple.C
                    WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTREF execSaxParserSimple.ref)
endif(xml)

ROOTTEST_ADD_TEST(PolyMarker
                  MACRO runPolyMarker.C
                  OUTREF PolyMarker.ref)

ROOTTEST_ADD_TEST(enginexml
                  MACRO runenginexml.C
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTREF enginexml.ref)

ROOTTEST_ADD_TEST(XmlDir
                  MACRO runXmlDir.C
                  OUTREF XmlDir.ref)

ROOTTEST_GENERATE_DICTIONARY(libXmlTestDictionaries
                  test_classes.h
                  LINKDEF LinkDef.h
                  NO_ROOTMAP NO_CXXMODULE
                  FIXTURES_SETUP xmlTestDict_fixture)

ROOTTEST_ADD_TEST(basicxml
                  MACRO runbasicxml.C
                  OUTREF basicxml.ref
                  FIXTURES_REQUIRED xmlTestDict_fixture)

ROOTTEST_ADD_TEST(filexml
                  MACRO runfilexml.C
                  OUTREF filexml.ref
                  FIXTURES_REQUIRED xmlTestDict_fixture)

