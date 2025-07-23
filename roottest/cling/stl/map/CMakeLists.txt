ROOTTEST_GENERATE_DICTIONARY(simpleDict testing.h NO_CXXMODULE NO_ROOTMAP LINKDEF LinkDef.h
                       FIXTURES_SETUP cling-stl-map-testing)

ROOTTEST_ADD_TEST(simpleTest
                  MACRO testing.C
                  ROOTEXE_OPTS -e "(void)gSystem->Load(\"simpleDict\")" 
                  OUTREF simpleTest.ref
                  FIXTURES_REQUIRED cling-stl-map-testing
                  LABEL roottest cling)

ROOTTEST_COMPILE_MACRO(MyOpClass.C
                       FIXTURES_SETUP cling-stl-map-MyOpClass-fixtures)

ROOTTEST_COMPILE_MACRO(MyClass.C
                       FIXTURES_SETUP cling-stl-map-MyClass-fixtures)

ROOTTEST_COMPILE_MACRO(stringMapLoad.cxx
                       FIXTURES_SETUP cling-stl-map-stringMapLoad-fixtures)

ROOTTEST_ADD_TEST(badOperator
                  MACRO badOperator.C
                  OUTREF badOperatorTest.ref
                  FIXTURES_REQUIRED cling-stl-map-MyOpClass-fixtures
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(stringMap
                  MACRO stringMap.cxx
                  OUTREF stringMapTest.ref
                  FIXTURES_REQUIRED cling-stl-map-stringMapLoad-fixtures
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(badstringMap
                  MACRO badstringMap.cxx
                  OUTREF badstringMapTest.ref
                  FIXTURES_REQUIRED cling-stl-map-stringMapLoad-fixtures cling-stl-map-MyClass-fixtures
                  LABELS roottest regression cling)
