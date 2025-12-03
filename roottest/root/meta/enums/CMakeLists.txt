ROOTTEST_GENERATE_REFLEX_DICTIONARY(fclassVal
                                    fclassVal.h
                                    SELECTION fclassVal_selection.xml
                                    FIXTURES_SETUP root-meta-enums-fclassVal-fixture)

ROOTTEST_ADD_TEST(fclassVal
                  MACRO execfclassVal.C
                  OUTREF execfclassVal.ref
                  FIXTURES_REQUIRED root-meta-enums-fclassVal-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(qualifiedNames
                                    qualifiedNames.h
                                    SELECTION qualifiedNames_selection.xml
                                    FIXTURES_SETUP root-meta-enums-qualifiedNames-fixture)

ROOTTEST_ADD_TEST(qualifiedNames
                  MACRO execQualifiedNames.C
                  OUTREF execQualifiedNames.ref
                  FIXTURES_REQUIRED root-meta-enums-qualifiedNames-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(enumsTestClasses
                                    enumsTestClasses.h
                                    SELECTION enumsTestClasses_selection.xml
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-meta-enums-enumsTestClasses-fixture)

ROOTTEST_ADD_TEST(execEnumsTest
                  MACRO execEnumsTest.C
                  OUTREF execEnumsTest.ref
                  FIXTURES_REQUIRED root-meta-enums-enumsTestClasses-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(tEnumGetEnumClasses
                                    tEnumGetEnumClasses.h
                                    SELECTION tEnumGetEnumClasses_selection.xml
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-meta-enums-tEnumGetEnumClasses-fixture)

ROOTTEST_ADD_TEST(execTEnumGetEnum
                  MACRO execTEnumGetEnum.C
                  OUTREF execTEnumGetEnum.ref
                  FIXTURES_REQUIRED root-meta-enums-tEnumGetEnumClasses-fixture)


ROOTTEST_GENERATE_REFLEX_DICTIONARY(enumsInNamespaces
                  enumsInNamespaces.h
                  SELECTION enumsInNamespaces_selection.xml
                  NO_ROOTMAP
                  FIXTURES_SETUP root-meta-enums-enumsInNamespaces-fixture)

ROOTTEST_ADD_TEST(enumsInNamespaces
                  MACRO execenumsInNamespaces.C
                  OUTREF execenumsInNamespaces.ref
                  FIXTURES_REQUIRED root-meta-enums-enumsInNamespaces-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(enumsInNamespaces2
                  enumsInNamespaces2.h
                  SELECTION enumsInNamespaces2_selection.xml
                  NO_ROOTMAP
                  FIXTURES_SETUP root-meta-enums-enumsInNamespaces2-fixture)

ROOTTEST_ADD_TEST(enumsInNamespaces2
                  MACRO execenumsInNamespaces2.C
                  OUTREF execenumsInNamespaces2.ref
                  FIXTURES_REQUIRED root-meta-enums-enumsInNamespaces2-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(duplicateEnums
                  duplicateEnums.h
                  SELECTION duplicateEnums_selection.xml
                  NO_ROOTMAP
                  FIXTURES_SETUP root-meta-enums-duplicateEnums-fixture)

ROOTTEST_ADD_TEST(duplicateEnums
                  MACRO execduplicateEnums.C
                  OUTREF execduplicateEnums.ref
                  FIXTURES_REQUIRED root-meta-enums-duplicateEnums-fixture)

ROOTTEST_GENERATE_DICTIONARY(libenumSize
                  enumSize.h
                  LINKDEF enumSizeLinkDef.h
                  NO_ROOTMAP NO_CXXMODULE
                  FIXTURES_SETUP root-meta-enums-libenumSize-fixture)

ROOTTEST_ADD_TEST(enumSize
                  MACRO execenumSize.C
                  OUTREF execenumSize.ref
                  FIXTURES_REQUIRED root-meta-enums-libenumSize-fixture)

ROOTTEST_ADD_TEST(testUsingEnum.cxx
                  MACRO test_usingenum.cxx
                  OUTREF test_usingenum.ref)
