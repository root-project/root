ROOTTEST_GENERATE_REFLEX_DICTIONARY(enumsTestClasses
                                    enumsTestClasses.h
                                    SELECTION enumsTestClasses_selection.xml
                                    NO_ROOTMAP)

ROOTTEST_ADD_TEST(roottest-root-meta-enums-execEnumsTest
                  MACRO execEnumsTest.C
                  OUTREF execEnumsTest.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(tEnumGetEnumClasses
                                    tEnumGetEnumClasses.h
                                    SELECTION tEnumGetEnumClasses_selection.xml
                                    NO_ROOTMAP)

ROOTTEST_ADD_TEST(roottest-root-meta-enums-execTEnumGetEnum
                  MACRO execTEnumGetEnum.C
                  OUTREF execTEnumGetEnum.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

