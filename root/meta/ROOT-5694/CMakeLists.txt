ROOTTEST_GENERATE_REFLEX_DICTIONARY(One One.h SELECTION One_selection.xml NO_ROOTMAP)
set(depends ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(Two Two.h SELECTION Two_selection.xml NO_ROOTMAP)
set(depends ${depends} ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TEST(execLoadLibs
                  MACRO execLoadLibs.C
                  OUTREF execLoadLibs.oref
                  ERRREF execLoadLibs.eref
                  DEPENDS ${depends})
