ROOTTEST_ADD_TEST(arraysize
                  MACRO runarraysize.C
                  OUTREF arraysize.ref)

ROOTTEST_ADD_TEST(ReadMiniObjLeaf
                  MACRO execReadMiniObjLeaf.C
                  COPY_TO_BUILDDIR miniObjTree.root
                  OUTREF execReadMiniObjLeaf.ref)
