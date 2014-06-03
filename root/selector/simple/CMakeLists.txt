add_definitions(
  -DClingWorkAroundMissingUnloading
  -DClingWorkAroundJITandInline 
  -DClingWorkAroundBrokenUnnamedReturn
  -DClingWorkAroundMissingDynamicScope
)

ROOTTEST_ADD_TEST(roottest-root-selector-simple-runtestLoadingSelector
                  MACRO runtestLoadingSelector.C
                  OUTREF testLoadingSelector.ref
                  OUTCNV LoadingSelector_convert.sh
                  DEPENDS testSel.C testSelector.C)
