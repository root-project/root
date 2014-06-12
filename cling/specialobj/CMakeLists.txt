# Add define to fix tests.
add_definitions(-DClingWorkAroundMissingDynamicScope)

ROOTTEST_ADD_TEST(roottest-cling-specialobj-assertGPad
                  MACRO assertGPad.C
                  OUTREF assertGPad.ref
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(AthIndex.h)
set(depends ${COMPILE_MACRO_TEST})

ROOTTEST_ADD_TEST(roottest-cling-specialobj-stlwrite
                  MACRO write.C
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(T0Result.h BUILDLIB AthIndex_h${libsuffix})
set(depends ${depends} ${COMPILE_MACRO_TEST})

ROOTTEST_COMPILE_MACRO(calib.h)
set(depends ${depends} ${COMPILE_MACRO_TEST})

ROOTTEST_ADD_TEST(roottest-cling-specialobj-argtwice
                  MACRO argeval.C
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-specialobj-stlProxies
                  MACRO runstlProxies.C
                  OUTREF stlProxies.ref
                  DEPENDS ${depends} roottest-cling-specialobj-stlwrite 
                  LABELS roottest regression cling)
