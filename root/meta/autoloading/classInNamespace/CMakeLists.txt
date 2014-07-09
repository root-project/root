ROOTTEST_GENERATE_REFLEX_DICTIONARY(classInNamespace classInNamespace.h
                                    SELECTION classInNamespace_selection.xml
                                    LIBNAME libclassInNamespace_dictrflx)

ROOTTEST_ADD_AUTOMACROS(DEPENDS ${GENERATE_REFLEX_TEST})
