ROOTTEST_GENERATE_REFLEX_DICTIONARY(classInNamespace classInNamespace.h
                                    SELECTION classInNamespace_selection.xml
                                    LIBNAME libclassInNamespace_dictrflx
                                    NO_ROOTMAP)

#---Copy from source to binary some of the files
configure_file(classInNamespace.rootmap . COPYONLY)

ROOTTEST_ADD_AUTOMACROS(DEPENDS ${GENERATE_REFLEX_TEST})
