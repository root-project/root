add_definitions(
  -DClingWorkAroundAutoParseUsingNamespace
  -DClingWorkAroundMissingDynamicScope
)

ROOTTEST_COMPILE_MACRO(TmpltInt0.C)
set(depends ${COMPILE_MACRO_TEST})

ROOTTEST_COMPILE_MACRO(TmpltInt1.C)
set(depends ${depends} ${COMPILE_MACRO_TEST})

ROOTTEST_COMPILE_MACRO(TmpltFloat.C)
set(depends ${depends} ${COMPILE_MACRO_TEST})

ROOTTEST_COMPILE_MACRO(TmpltNoSpec.C)
set(depends ${depends} ${COMPILE_MACRO_TEST})

ROOTTEST_ADD_AUTOMACROS(DEPENDS ${depends})
