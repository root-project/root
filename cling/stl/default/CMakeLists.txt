ROOTTEST_GENERATE_DICTIONARY(dict LINKDEF mylist.C LinkDef.h)

ROOTTEST_ADD_TEST(roottest-cling-stl-default-defaultArgTest
                  MACRO build.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-stl-default-VectorSort
                  MACRO VectorSort.C
                  OUTREF VectorSort.ref
                  OUTCNV VectorSort_convert.sh
                  LABELS roottest regression cling)
