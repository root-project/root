ROOTTEST_ADD_TEST(assertGPad
                  MACRO assertGPad.C
                  OUTREF assertGPad.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(stlwrite
                  MACRO write.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(argtwice
                  MACRO argeval.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(stlProxies
                  MACRO runstlProxies.C
                  OUTREF stlProxies.ref
                  DEPENDS roottest-cling-specialobj-stlwrite 
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runf02
                  MACRO runf02.C
                  OUTCNVCMD sed -e s,input_line_[0-9]*,input_line_FILTERED,g
                  OUTREF f02.ref
                  LABELS roottest regression cling)
