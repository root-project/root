ROOTTEST_GENERATE_DICTIONARY(libDataObject dataObject.hxx 
                             LINKDEF LinkDef.h
                             NO_CXXMODULE
                             FIXTURES_SETUP avoidParsingLibrary)

ROOTTEST_ADD_TEST(avoidParsingSpace
                  MACRO runAvoidParsing.C
                  MACROARG true
                  OUTREF avoidParsing.ref
                  FIXTURES_REQUIRED avoidParsingLibrary)

ROOTTEST_ADD_TEST(avoidParsingNoSpace
                  MACRO runAvoidParsing.C
                  MACROARG false
                  OUTREF avoidParsing.ref
                  FIXTURES_REQUIRED avoidParsingLibrary)