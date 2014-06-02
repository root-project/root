add_definitions(
  -DClingWorkAroundMissingAutoLoadingForTemplates
)

ROOTTEST_COMPILE_MACRO(spaces.C)

ROOTTEST_ADD_TEST(roottest-root-meta-rootmap-runspaces
                  MACRO runspaces.C
                  OUTREF spaces.ref
                  DEPENDS ${COMPILE_MACRO_TEST})
