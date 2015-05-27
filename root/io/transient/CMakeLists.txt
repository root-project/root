ROOTTEST_ADD_TEST(roottest-root-io-transient-execAtomTransient
                  MACRO execAtomTransient.cxx+
                  OUTREF execAtomTransient.ref)

ROOTTEST_ADD_TEST(roottest-root-io-transient-execDoxygenTransient
                  MACRO execDoxygenTransient.cxx+
                  OUTREF execDoxygenTransient.ref)

ROOTTEST_ADD_TEST(roottest-root-io-transient-execDoxygenTransientInterp
                  MACRO execDoxygenTransientInterp.C
                  OUTREF execDoxygenTransientInterp.ref)


ROOTTEST_ADD_TEST(roottest-root-io-transient-execOld
                  MACRO execOld.cxx+
                  OUTREF execOld.ref)

ROOTTEST_ADD_TEST(roottest-root-io-transient-execNew
                  MACRO execNew.cxx+
                  OUTREF execNew.ref
                  DEPENDS roottest-root-io-transient-execOld)

ROOTTEST_GENERATE_REFLEX_DICTIONARY( transientNoctor transientNoctor.h
                                     SELECTION transientNoctor_selection.xml
                                     NO_ROOTMAP)

ROOTTEST_ADD_TEST(roottest-root-io-transient-execTransientNoctor
                  MACRO execTransientNoctor.C
                  OUTREF execTransientNoctor.ref
                  DEPENDS ${GENERATE_REFLEX_TEST})

ROOTTEST_ADD_TESTDIRS()
