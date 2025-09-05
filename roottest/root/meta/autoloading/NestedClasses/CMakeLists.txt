ROOTTEST_GENERATE_REFLEX_DICTIONARY(nestedTemplate
                                    nestedTemplate.h
                                    SELECTION sel1.xml
                                    FIXTURES_SETUP root-meta-autoloading-NestedClasses-dic1-fixture)

if(NOT MSVC OR win_broken_tests)
    ROOTTEST_GENERATE_REFLEX_DICTIONARY(nestedTemplateClone
                                        nestedTemplate_clone.h
                                        SELECTION sel2.xml
                                        FIXTURES_SETUP root-meta-autoloading-NestedClasses-dic1-fixture)
endif()

ROOTTEST_ADD_TEST(nestedTemplate
                  MACRO execnestedTemplate.C
                  OUTREF nestedTemplate.ref
                  FIXTURES_REQUIRED root-meta-autoloading-NestedClasses-dic1-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(nestedClasses
                                    nestedClasses.h
                                    SELECTION nestedClasses_selection.xml
                                    FIXTURES_SETUP root-meta-autoloading-NestedClasses-dic2-fixture)
ROOTTEST_ADD_TEST(nestedClasses
                  MACRO execNestedClasses.C
                  OUTREF execNestedClasses${ref_suffix}
                  FIXTURES_REQUIRED root-meta-autoloading-NestedClasses-dic2-fixture)
