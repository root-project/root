# Add defines to fix tests.
add_definitions(
  -DClingWorkAroundMissingDynamicScope
  -DClingWorkAroundUnnamedInclude
  -DClingWorkAroundMissingSmartInclude
)

ROOTTEST_COMPILE_MACRO(t.h)

ROOTTEST_ADD_TEST(roottest-cling-functionTemplate-cintrun
                  MACRO cintrun.C
                  OUTREF cintrun.ref-cling
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(MyClassReferenceUse.C)

ROOTTEST_ADD_TEST(roottest-cling-functionTemplate-runreferenceuse
                  MACRO runreferenceUse.C
                  OUTREF referenceUse.ref
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-functionTemplate-testcint
                  MACRO testcint.py
                  OUTREF pythoncintrun.ref
                  OUTCNVCMD grep -v "just a comment"
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-functionTemplate-testreflex
                  MACRO testreflex.py
                  OUTREF pythonreflexrun.ref
                  OUTCNVCMD grep -v "just a comment"
                  LABELS roottest regression cling)
