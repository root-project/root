ROOTTEST_ADD_TESTDIRS()

ROOTTEST_GENERATE_REFLEX_DICTIONARY(typedefSelection typedefSelection.h SELECTION typedefSelection_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-typedefSelection
                  MACRO execTypedefSelection.C
                  OUTREF execTypedefSelection.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_GENERATE_REFLEX_DICTIONARY(stlPatternSelection stlPatternSelection.h SELECTION stlPatternSelection_selection.xml)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-stlPatternSelection
                  MACRO execstlPatternSelection.C
                  OUTREF execstlPatternSelection.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})
