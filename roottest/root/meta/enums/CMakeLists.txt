ROOTTEST_GENERATE_REFLEX_DICTIONARY(fclassVal
                                    fclassVal.h
                                    SELECTION fclassVal_selection.xml)

ROOTTEST_ADD_TEST(fclassVal
                  MACRO execfclassVal.C
                  OUTREF execfclassVal.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(qualifiedNames
                                    qualifiedNames.h
                                    SELECTION qualifiedNames_selection.xml)

ROOTTEST_ADD_TEST(qualifiedNames
                  MACRO execQualifiedNames.C
                  OUTREF execQualifiedNames.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(enumsTestClasses
                                    enumsTestClasses.h
                                    SELECTION enumsTestClasses_selection.xml
                                    NO_ROOTMAP)

ROOTTEST_ADD_TEST(execEnumsTest
                  MACRO execEnumsTest.C
                  OUTREF execEnumsTest.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(tEnumGetEnumClasses
                                    tEnumGetEnumClasses.h
                                    SELECTION tEnumGetEnumClasses_selection.xml
                                    NO_ROOTMAP)

ROOTTEST_ADD_TEST(execTEnumGetEnum
                  MACRO execTEnumGetEnum.C
                  OUTREF execTEnumGetEnum.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})


ROOTTEST_GENERATE_REFLEX_DICTIONARY(enumsInNamespaces
                  enumsInNamespaces.h
                  SELECTION enumsInNamespaces_selection.xml
                  NO_ROOTMAP)

ROOTTEST_ADD_TEST(enumsInNamespaces
                  MACRO execenumsInNamespaces.C
                  OUTREF execenumsInNamespaces.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(enumsInNamespaces2
                  enumsInNamespaces2.h
                  SELECTION enumsInNamespaces2_selection.xml
                  NO_ROOTMAP)

ROOTTEST_ADD_TEST(enumsInNamespaces2
                  MACRO execenumsInNamespaces2.C
                  OUTREF execenumsInNamespaces2.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(duplicateEnums
                  duplicateEnums.h
                  SELECTION duplicateEnums_selection.xml
                  NO_ROOTMAP)

ROOTTEST_ADD_TEST(duplicateEnums
                  MACRO execduplicateEnums.C
                  OUTREF execduplicateEnums.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_DICTIONARY(libenumSize
                  enumSize.h
                  LINKDEF enumSizeLinkDef.h
                  NO_ROOTMAP NO_CXXMODULE)

ROOTTEST_ADD_TEST(enumSize
                  MACRO execenumSize.C
                  OUTREF execenumSize.ref
                  DEPENDS ${GENERATE_DICTIONARY_TEST})

ROOTTEST_ADD_TEST(testUsingEnum.cxx
                  MACRO test_usingenum.cxx
                  OUTREF test_usingenum.ref)
