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
if(MSVC)
    set(NESTEDCLASSES_OUTREF execNestedClasses_win32.ref)
else()
    set(NESTEDCLASSES_OUTREF execNestedClasses.ref)
endif()
ROOTTEST_ADD_TEST(nestedClasses
                  MACRO execNestedClasses.C
                  OUTREF ${NESTEDCLASSES_OUTREF}
                  DEPENDS ${GENERATE_REFLEX_TEST})
