ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes
                                    classes.h
                                    SELECTION classes_selection.xml
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-meta-autoloading-templates-classes-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes2
                                    classes2.h
                                    SELECTION classes2_selection.xml
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-meta-autoloading-templates-classes2-fixture)

ROOTTEST_ADD_TEST(TemplateAutoloading
                  MACRO execTemplateAutoloading.C
                  COPY_TO_BUILDDIR theRootmap.rootmap
                  OUTREF execTemplateAutoloading.ref
                  FIXTURES_REQUIRED root-meta-autoloading-templates-classes-fixture
                                    root-meta-autoloading-templates-classes2-fixture)

