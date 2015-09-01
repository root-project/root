ROOTTEST_GENERATE_DICTIONARY(dict.C testing.C LINKDEF LinkDef.h)

ROOTTEST_ADD_TEST(roottest-cling-stl-map-badOperator
                  MACRO badOperator.C
                  OUTREF badOperatorTest.ref
                  DEPENDS MyOpClass.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-stl-map-stringMap
                  MACRO stringMap.cxx
                  OUTREF stringMapTest.ref
                  DEPENDS stringMapLoad.cxx
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-stl-map-badstringMap
                  MACRO badstringMap.cxx
                  OUTREF badstringMapTest.ref
                  DEPENDS MyClass.C
                  LABELS roottest regression cling)
