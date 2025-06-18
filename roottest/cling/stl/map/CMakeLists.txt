ROOTTEST_GENERATE_DICTIONARY(dict.C testing.C LINKDEF LinkDef.h)

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
