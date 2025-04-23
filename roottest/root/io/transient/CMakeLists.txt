ROOTTEST_ADD_TEST(execAtomTransient
                  MACRO execAtomTransient.cxx+
                  OUTREF execAtomTransient.ref)

ROOTTEST_ADD_TEST(execDoxygenTransient
                  MACRO execDoxygenTransient.cxx+
                  OUTREF execDoxygenTransient.ref)

ROOTTEST_ADD_TEST(execDoxygenTransientInterp
                  MACRO execDoxygenTransientInterp.C
                  OUTREF execDoxygenTransientInterp.ref)


ROOTTEST_ADD_TEST(execOld
                  MACRO execOld.cxx+
                  OUTREF execOld.ref)

ROOTTEST_ADD_TEST(execNew
                  MACRO execNew.cxx+
                  OUTREF execNew.ref
                  DEPENDS execOld)

ROOTTEST_GENERATE_REFLEX_DICTIONARY( transientNoctor transientNoctor.h
                                     SELECTION transientNoctor_selection.xml
                                     NO_ROOTMAP)

ROOTTEST_ADD_TEST(execTransientNoctor
                  MACRO execTransientNoctor.C
                  OUTREF execTransientNoctor.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TESTDIRS()
