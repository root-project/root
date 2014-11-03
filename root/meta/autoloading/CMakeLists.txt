ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes classes.h SELECTION classes_selection.xml NO_ROOTMAP)
set(depends ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(TriggerTypeDefs TriggerTypeDefs.h SELECTION TriggerTypeDefs_selection.xml)
set(depends ${depends} ${GENERATE_REFLEX_TEST})

#---Copy from source to binary some of the files
configure_file(exectypeDefAutoloading.rootmap . COPYONLY)
configure_file(aRootmap.rootmap . COPYONLY)

#--With ACLiC---------------------------------------------------------
set(execTypeinfo-suffix +)

ROOTTEST_ADD_AUTOMACROS(DEPENDS ${GENERATE_REFLEX_TEST})
ROOTTEST_ADD_TESTDIRS()
