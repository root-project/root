add_definitions(
  -DClingWorkAroundMissingUnloading
  -DClingWorkAroundJITandInline 
  -DClingWorkAroundBrokenUnnamedReturn
  -DClingWorkAroundMissingDynamicScope
)

ROOTTEST_COMPILE_MACRO(testSel.C)
set(depends ${COMPILE_MACRO_TEST})

ROOTTEST_COMPILE_MACRO(testSelector.C)
set(depends ${depends} ${COMPILE_MACRO_TEST})

ROOTTEST_ADD_TEST(roottest-root-selector-simple-runtestLoadingSelector
                  MACRO runtestLoadingSelector.C
                  OUTREF testLoadingSelector.ref
                  OUTCNV LoadingSelector_convert.sh
                  DEPENDS ${depends})
