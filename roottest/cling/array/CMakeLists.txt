ROOTTEST_ADD_TEST(runarray1
                  COPY_TO_BUILDDIR array1.C
                  MACRO runarray1test.C
                  OUTREF array1test.ref
                  OUTCNV array1test_convert.sh
                  LABELS roottest regression cling)
