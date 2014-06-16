ROOTTEST_GENERATE_DICTIONARY(dict LINKDEF mylist.C LinkDef.h)

ROOTTEST_ADD_TEST(roottest-cling-stl-default-defaultArgTest
                  MACRO build.C
                  LABELS roottest regression cling)

if(ClingWorkAroundJITandInline)
  set(testcint TEST_OUTREF_CINTSPECIFIC) 
else()
  set(testcint "")
endif()

ROOTTEST_ADD_TEST(roottest-cling-stl-default-VectorSort
                  MACRO VectorSort.C
                  ${testcint}
                  OUTREF VectorSort.ref
                  OUTCNV VectorSort_convert.sh
                  LABELS roottest regression cling)
