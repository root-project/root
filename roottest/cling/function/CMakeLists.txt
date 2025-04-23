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
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(runstaticfunc
                  MACRO runstaticfunc.C
                  OUTREF_CINTSPECIFIC staticfunc.ref
                  # OUTCNV staticfunc_convert.sh
                  ${WILLFAIL_ON_WIN32}
                  LABELS roottest regression cling)

if(NOT (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES arm64) OR M1_BROKEN_TESTS)
  ROOTTEST_ADD_TEST(trybody
                    MACRO trybody.C
                    OUTREF trybody.ref
                    ${WILLFAIL_ON_WIN32}
                    LABELS roottest regression cling)
endif()

ROOTTEST_COMPILE_MACRO(refFunction.C)
ROOTTEST_COMPILE_MACRO(refClasses.cxx)
