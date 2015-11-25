ROOTTEST_ADD_TEST(runtestLoadingSelector
                  MACRO runtestLoadingSelector.C
                  OUTREF testLoadingSelector.ref
                  OUTCNV LoadingSelector_convert.sh
                  DEPENDS testSel.C testSelector.C)
