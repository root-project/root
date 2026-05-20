ROOTTEST_ADD_TEST(InterpActualClass
                  MACRO execInterpActualClass.C
                  OUTREF execInterpActualClass.ref)

ROOTTEST_ADD_TEST(TestTClassGlobalIsA
                  MACRO TestTClassGlobalIsA.C+
                  OUTREF TestTClassGlobalIsA.ref)
