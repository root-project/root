ROOTTEST_ADD_TEST(runoverloadResolution
                  MACRO runoverloadResolution.C
                  OUTREF overloadResolution.ref
                  DEPENDS TheClass.h
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(rundefaultParams
                  MACRO rundefaultParams.C
                  OUTREF defaultParams.ref
                  WILLFAIL
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(execrefDefaultParams
                  MACRO execrefDefaultParams.C
                  OUTREF execrefDefaultParams.ref
                  WILLFAIL
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runlongArgTest
                  MACRO runlongArgTest.C
                  OUTREF longArgTest.ref
                  WILLFAIL
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runstaticfunc
                  MACRO runstaticfunc.C
                  OUTREF_CINTSPECIFIC staticfunc.ref
                  # OUTCNV staticfunc_convert.sh
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(refFunction.C)
ROOTTEST_COMPILE_MACRO(refClasses.cxx)
