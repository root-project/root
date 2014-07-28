ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes classes.h SELECTION classes_selection.xml NO_ROOTMAP)

ROOTTEST_ADD_AUTOMACROS(DEPENDS ${GENERATE_REFLEX_TEST}
                        COPY_TO_BUILDDIR exectypeDefAutoloading.rootmap aRootmap.rootmap)
ROOTTEST_ADD_TESTDIRS()
