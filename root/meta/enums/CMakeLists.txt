ROOTTEST_GENERATE_REFLEX_DICTIONARY(enumsTestClasses
                                    enumsTestClasses.h
                                    SELECTION enumsTestClasses_selection.xml
                                    NO_ROOTMAP)

ROOTTEST_ADD_TEST(roottest-root-meta-enums-execEnumsTest
                  MACRO execEnumsTest.C
                  OUTREF execEnumsTest.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
