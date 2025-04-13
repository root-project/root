ROOTTEST_GENERATE_REFLEX_DICTIONARY(nestedTemplate
                                    nestedTemplate.h
                                    SELECTION sel1.xml)

if(NOT MSVC OR win_broken_tests)
    ROOTTEST_GENERATE_REFLEX_DICTIONARY(nestedTemplateClone
                                        nestedTemplate_clone.h
                                        SELECTION sel2.xml)
endif()

ROOTTEST_ADD_TEST(nestedTemplate
                  MACRO execnestedTemplate.C
                  OUTREF nestedTemplate.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(nestedClasses
                                    nestedClasses.h
                                    SELECTION nestedClasses_selection.xml)
ROOTTEST_ADD_TEST(nestedClasses
                  MACRO execNestedClasses.C
                  OUTREF execNestedClasses${ref_suffix}
                  DEPENDS ${GENERATE_REFLEX_TEST})
