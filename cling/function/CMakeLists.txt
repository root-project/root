ROOTTEST_COMPILE_MACRO(TheClass.h)

ROOTTEST_ADD_TEST(roottest-cling-function-runoverloadResolution
                  MACRO runoverloadResolution.C
                  OUTREF overloadResolution.ref
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-function-rundefaultParams
                  MACRO rundefaultParams.C
                  OUTREF defaultParams.ref
                  WILLFAIL
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-function-execrefDefaultParams
                  MACRO execrefDefaultParams.C
                  OUTREF execrefDefaultParams.ref
                  WILLFAIL
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-function-runlongArgTest
                  MACRO runlongArgTest.C
                  OUTREF longArgTest.ref
                  WILLFAIL
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-function-runstaticfunc
                  MACRO runstaticfunc.C
                  OUTREF staticfunc.ref-cling
                  OUTCNV staticfunc_convert.sh
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(refFunction.C)
ROOTTEST_COMPILE_MACRO(refClasses.cxx)
