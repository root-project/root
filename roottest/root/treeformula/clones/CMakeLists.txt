ROOTTEST_ADD_TEST(nested-2
                  COPY_TO_BUILDDIR digi.root
                  MACRO digiRun.C
                  OUTREF nestedClones2.ref)

ROOTTEST_ADD_TEST(nested-3
                  COPY_TO_BUILDDIR nesting3.root
                  MACRO runNestedClones3.C
                  OUTREF nestedClones3.ref)
