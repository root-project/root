# Add defines to fix tests.
add_definitions(
  -DClingWorkAroundMissingDynamicScope
  -DClingWorkAroundUnloadingIOSTREAM
)

ROOT_COMPILE_MACRO(longlong.C)
ROOT_COMPILE_MACRO(Singleton.h)

#ROOT_BUILD_DICT(masterDict.C FILES master.h masterLinkDef.h)
#ROOT_BUILD_DICT(slaveDict.C FILES slave.h slaveLinkDef.h)
#ROOT_BUILD_DICT(forwardDict.C FILES forward.C linkdef.h)
#ROOT_BUILD_DICT(constructorDict.C FILES constructor.hh linkdef.h)
#ROOT_BUILD_DICT(typenameTestDict.C FILES typenameTest.C linkdef.h)

ROOT_ADD_ROOTTEST(cling-template-templateSingleton MACRO runtemplateSingleton.C OUTREF templateSingleton.ref)
ROOT_ADD_ROOTTEST(cling-template-longlong MACRO runlonglong.C OUTREF longlong.ref)
ROOT_ADD_ROOTTEST(cling-template-usingtemplate MACRO runusingtemplate.C OUTREF usingtemplate.ref)
ROOT_ADD_ROOTTEST(cling-template-operator1-interpreted MACRO smalltest.C)
ROOT_ADD_ROOTTEST(cling-template-operator1-compiled MACRO smalltest.C+)
ROOT_ADD_ROOTTEST(cling-template-templatefriend MACRO runtemplatefriend.C OUTREF templatefriend.ref)
