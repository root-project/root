# ROOTTEST_GENERATE_REFLEX_DICTIONARY(clingFwdDeclExample clingFwdDeclExample.h SELECTION clingFwdDeclExample_selection.xml NO_ROOTMAP)

ROOTTEST_ADD_TEST(clingFwdDeclExample
                  COPY_TO_BUILDDIR clingFwdDeclExample.h clingFwdDeclExample_selection.xml A.h checkDict
                  COMMAND checkDict
                  OUTREF clingFwdDeclExample.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})


ROOTTEST_GENERATE_REFLEX_DICTIONARY(fwdDeclarations fwdDeclarations.h SELECTION fwdDeclarations_selection.xml NO_ROOTMAP)

ROOTTEST_ADD_TEST(fwdDeclarations
                  MACRO execfwdDeclarations.C
                  OUTREF execfwdDeclarations.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TESTDIRS()
