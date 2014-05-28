ROOTTEST_COMPILE_MACRO(vclass.h)

ROOTTEST_ADD_TEST(roottest-cling-typedef-runptrptr
                  MACRO runptrptr.C
                  OUTREF ptrptr.ref
                  OUTCNVCMD grep ptrptr.C
                  LABELS roottest regression cling)

ROOTTEST_COMPILE_MACRO(testclass.h)
ROOTTEST_ADD_TEST(roottest-cling-typedef-testclass
                  MACRO -e "gSystem->Load(\"testclass_h\")"
                  OUTREF typedefAndScope.ref
                  DEPENDS ${COMPILE_MACRO_TEST}
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-typedef-assertTypedefIter
                  MACRO assertTypedefIter.C
                  OUTREF assertTypedefIter.ref
                  LABELS roottest regression cling)

ROOTTEST_ADD_TEST(roottest-cling-typedef-assertFuncArray
                  MACRO assertFuncArray.cxx+
                  OUTREF assertFuncArray.ref
                  LABELS roottest regression cling)
