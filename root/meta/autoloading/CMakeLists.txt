ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes classes.h SELECTION classes_selection.xml NO_ROOTMAP)

#---Copy from source to binary some of the files
configure_file(exectypeDefAutoloading.rootmap . COPYONLY)
configure_file(aRootmap.rootmap . COPYONLY)

ROOTTEST_ADD_AUTOMACROS(DEPENDS ${GENERATE_REFLEX_TEST})
ROOTTEST_ADD_TESTDIRS()
