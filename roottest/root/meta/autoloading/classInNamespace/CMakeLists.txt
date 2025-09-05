ROOTTEST_GENERATE_REFLEX_DICTIONARY(classInNamespace
                                    classInNamespace.h
                                    SELECTION classInNamespace_selection.xml
                                    LIBNAME libclassInNamespace_dictrflx
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-meta-autoloading-classInNamespace-dict-fixture)

ROOTTEST_ADD_TEST(ClassInNamespace
                  MACRO execClassInNamespace.C
                  OUTREF execClassInNamespace.ref
                  COPY_TO_BUILDDIR classInNamespace.rootmap
                  FIXTURES_REQUIRED root-meta-autoloading-classInNamespace-dict-fixture)