ROOTTEST_COMPILE_MACRO(main.C
                       FIXTURES_SETUP root-treeformula-morrison-func-fixture)

ROOTTEST_ADD_TEST(ArraySyntax
                  MACRO run.C
                  OUTREF ArraySyntax.ref
                  FIXTURES_REQUIRED root-treeformula-morrison-func-fixture)
