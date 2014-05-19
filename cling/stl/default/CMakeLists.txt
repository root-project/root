ROOTTEST_GENERATE_DICTIONARY(dict LINKDEF mylist.C LinkDef.h)

ROOT_ADD_ROOTTEST(cling-stl-default-defaultArgTest MACRO build.C)
ROOT_ADD_ROOTTEST(cling-stl-default-VectorSort MACRO VectorSort.C OUTREF VectorSort.ref OUTCNV VectorSort_convert.sh)
