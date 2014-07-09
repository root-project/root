ROOTTEST_GENERATE_REFLEX_DICTIONARY(classInNamespace classInNamespace.h
                                    SELECTION classInNamespace_selection.xml
                                    LIBNAME libclassInNamespace_dictrflx
                                    NO_ROOTMAP)

ROOTTEST_ADD_AUTOMACROS(DEPENDS ${GENERATE_REFLEX_TEST}
                        COPY_TO_BUILDDIR classInNamespace.rootmap)
