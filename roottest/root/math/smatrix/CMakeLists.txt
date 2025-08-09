ROOTTEST_ADD_TEST(Inversion
                  MACRO testInversion.cxx+
                  ERRREF testInversion.ref)

ROOTTEST_ADD_TEST(Kalman
                  MACRO testKalman.cxx+
                  ERRREF testKalman.ref)

ROOTTEST_ADD_TEST(Operations
                  MACRO testOperations.cxx+
                  ERRREF testOperations.ref)

ROOTTEST_ADD_TEST(SMatrix
                  MACRO testSMatrix.cxx+
                  ERRREF testSMatrix.ref)
