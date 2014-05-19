# Add defines to fix cling-function-staticfunc
#add_definitions(-DClingWorkAroundBrokenRecovery)

ROOT_COMPILE_MACRO(TheClass.h)

ROOT_ADD_ROOTTEST(cling-function-defaultParams MACRO rundefaultParams.C
                               OUTREF defaultParams.ref WILLFAIL)

ROOT_ADD_ROOTTEST(cling-function-longArgTest MACRO runlongArgTest.C
                               OUTREF longArgTest.ref WILLFAIL)

ROOT_ADD_ROOTTEST(cling-function-overloadResolution MACRO runoverloadResolution.C
                               OUTREF overloadResolution.ref)

ROOT_ADD_ROOTTEST(cling-function-staticfunc MACRO runstaticfunc.C
                               OUTREF staticfunc.ref-cling OUTCNV staticfunc_convert.sh)
