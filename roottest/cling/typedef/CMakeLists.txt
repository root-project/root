ROOTTEST_COMPILE_MACRO(vclass.h)

ROOTTEST_ADD_TEST(runptrptr
                  MACRO runptrptr.C
                  OUTREF_CINTSPECIFIC ptrptr.ref
                  OUTCNVCMD grep ptrptr.C
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(testclass
                  MACRO -e "gSystem->Load(\"testclass_h\")"
                  OUTREF typedefAndScope.ref
                  OUTCNVCMD grep -v \(int\)\ 0
                  DEPENDS testclass.h
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(assertTypedefIter
                  MACRO assertTypedefIter.C
                  OUTREF assertTypedefIter.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(assertFuncArray
                  MACRO assertFuncArray.cxx+
                  OUTREF assertFuncArray.ref
                  LABELS roottest regression cling)
