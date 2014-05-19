# Add define to fix tests.
add_definitions(-DClingWorkAroundMissingDynamicScope)

ROOT_COMPILE_MACRO(AthIndex.h)
ROOT_COMPILE_MACRO(T0Result.h BUILDLIB AthIndex_h${libsuffix})

ROOT_COMPILE_MACRO(calib.h)

ROOT_ADD_ROOTTEST(cling-specialobj-stlwrite   MACRO write.C)
ROOT_ADD_ROOTTEST(cling-specialobj-stlProxies MACRO runstlProxies.C OUTREF stlProxies.ref DEPENDS cling-specialobj-stlwrite)
ROOT_ADD_ROOTTEST(cling-specialobj-argtwice   MACRO argeval.C)
