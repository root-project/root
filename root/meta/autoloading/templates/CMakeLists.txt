ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes classes.h SELECTION classes_selection.xml)
set(depends ${GENERATE_REFLEX_TESET})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes2 classes2.h SELECTION classes2_selection.xml)
set(depends ${depends} ${GENERATE_REFLEX_TESET})

ROOTTEST_ADD_AUTOMACROS(DEPENDS ${depends} COPY_TO_BUILDDIR theRootmap.rootmap)

