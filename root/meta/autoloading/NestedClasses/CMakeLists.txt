ROOTTEST_GENERATE_REFLEX_DICTIONARY(nestedTemplate
                                    nestedTemplate.h
                                    SELECTION sel1.xml)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(nestedTemplateClone
                                    nestedTemplate_clone.h
                                    SELECTION sel2.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-autoloading-NestedClasses-nestedTemplate
                  MACRO execnestedTemplate.C
                  OUTREF nestedTemplate.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(nestedClasses
                                    nestedClasses.h
                                    SELECTION nestedClasses_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-autoloading-NestedClasses-execNestedClasses
                  MACRO execNestedClasses.C
                  OUTREF execNestedClasses.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
