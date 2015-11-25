ROOTTEST_GENERATE_DICTIONARY(dict mylist.C LINKDEF LinkDef.h)

ROOTTEST_ADD_TEST(defaultArgTest
                  COPY_TO_BUILDDIR mylist.C
                  MACRO build.C
                  LABELS roottest regression cling)

if(ClingWorkAroundJITandInline)
  set(testcint OUTREF_CINTSPECIFIC) 
else()
  set(testcint OUTREF)
endif()

ROOTTEST_ADD_TEST(VectorSort
                  MACRO VectorSort.C
                  ${testcint} VectorSort.ref
                  # OUTCNV VectorSort_convert.sh
                  LABELS roottest regression cling)
