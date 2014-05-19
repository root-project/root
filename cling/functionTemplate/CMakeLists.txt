# Add defines to fix tests.
add_definitions(
  -DClingWorkAroundMissingDynamicScope
  -DClingWorkAroundUnnamedInclude
  -DClingWorkAroundMissingSmartInclude
)

ROOT_COMPILE_MACRO(t.h)
ROOT_COMPILE_MACRO(MyClassReferenceUse.C)

ROOT_ADD_ROOTTEST(cling-functionTemplate-cintrun MACRO cintrun.C
                               OUTREF cintrun.ref-cling)

ROOT_ADD_ROOTTEST(cling-functionTemplate-referenceuse MACRO runreferenceUse.C
                               OUTREF referenceUse.ref)
