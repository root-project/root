add_definitions(
  -DClingWorkAroundAutoParseUsingNamespace
  -DClingWorkAroundMissingDynamicScope
)

ROOTTEST_ADD_AUTOMACROS(DEPENDS TmpltInt0.C TmpltInt1.C TmpltFloat.C TmpltNoSpec.C)
ROOTTEST_FIND_TESTDIRS()
