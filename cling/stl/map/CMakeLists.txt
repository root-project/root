add_definitions(
  -DClingWorkAroundMissingDynamicScope
  -DClingWorkAroundUnnamedInclude
)

ROOTTEST_GENERATE_DICTIONARY(dict.C LINKDEF testing.C LinkDef.h)

ROOTTEST_COMPILE_MACRO(MyOpClass.C)

ROOTTEST_ADD_TEST(roottest-cling-stl-map-badOperator
                  MACRO badOperator.C
                  OUTREF badOperatorTest.ref
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(stringMapLoad.cxx)

ROOTTEST_ADD_TEST(roottest-cling-stl-map-stringMap
                  MACRO stringMap.cxx
                  OUTREF stringMapTest.ref
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(MyClass.C)

ROOTTEST_ADD_TEST(roottest-cling-stl-map-badstringMap
                  MACRO badstringMap.cxx
                  OUTREF badstringMapTest.ref
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)
