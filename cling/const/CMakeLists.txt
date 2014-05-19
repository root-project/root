ROOT_ADD_ROOTTEST(cling-const-run1 MACRO runConst.C OUTREF good1.output)
ROOT_ADD_ROOTTEST(cling-const-run2 MACRO runConst.C+ OUTREF good2.output)
ROOT_ADD_ROOTTEST(cling-const-constCharTest MACRO ../../root/build.C\(\"constissue.h\"\))
