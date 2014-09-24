ROOTTEST_GENERATE_REFLEX_DICTIONARY(fwdDeclarations fwdDeclarations.h SELECTION fwdDeclarations_selection.xml NO_ROOTMAP)

ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-fwdDecl-fwdDeclarations
                  MACRO execfwdDeclarations.C
                  OUTREF execfwdDeclarations.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TESTDIRS()
