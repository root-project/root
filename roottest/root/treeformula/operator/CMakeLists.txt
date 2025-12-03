ROOTTEST_ADD_TEST(modulo
                  MACRO runmodulo.C
                  OUTREF modulo.ref)

ROOTTEST_ADD_TEST(ternary
                  MACRO runternary.C
                  OUTREF ternary.ref)

ROOTTEST_ADD_TEST(largebitwise
                  MACRO runlargebitwise.C
                  OUTREF largebitwise.ref)
