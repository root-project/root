ROOTTEST_ADD_TEST(roottest-cling-specialobj-assertGPad
                  MACRO assertGPad.C
                  OUTREF assertGPad.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-specialobj-stlwrite
                  MACRO write.C
                  DEPENDS AthIndex.h T0Result.h calib.h
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-specialobj-argtwice
                  MACRO argeval.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-specialobj-stlProxies
                  MACRO runstlProxies.C
                  OUTREF stlProxies.ref
                  DEPENDS roottest-cling-specialobj-stlwrite 
                  LABELS roottest regression cling)
