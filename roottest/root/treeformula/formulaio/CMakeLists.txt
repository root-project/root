ROOTTEST_ADD_TEST(old-formula
                  COPY_TO_BUILDDIR hsimple.31000.root
                  MACRO runoldFormulaTest.C
                  OUTREF oldFormulaTest.ref)
