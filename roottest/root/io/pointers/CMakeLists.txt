ROOTTEST_ADD_TEST(multiTree
                  MACRO runCanvases.C
                  OUTREF Canvases.ref)

ROOTTEST_ADD_TEST(restoreCanvas
                  MACRO restoreCanvas.C
                  OUTREF restoreCanvas.ref)
