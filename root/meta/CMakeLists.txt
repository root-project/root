add_definitions(
  -DClingWorkAroundAutoParseUsingNamespace
  -DClingWorkAroundMissingDynamicScope
)

ROOTTEST_GENERATE_EXECUTABLE(loadernotapp loadernotapp.cxx LIBRARIES Hist)
ROOTTEST_ADD_TEST(roottest-root-meta-loadernotapp
                  EXEC loadernotapp
                  OUTREF loadernotapp.ref
                  DEPENDS ${GENERATE_EXECUTABLE_TEST})

ROOTTEST_COMPILE_MACRO(fornamespace.C)

ROOTTEST_ADD_AUTOMACROS(DEPENDS TmpltInt0.C TmpltInt1.C TmpltFloat.C TmpltNoSpec.C ${GENERATE_MACRO_TEST})
ROOTTEST_FIND_TESTDIRS()
