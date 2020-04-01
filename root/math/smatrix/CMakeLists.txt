ROOTTEST_ADD_TEST(testInversion
                MACRO ${CMAKE_CURRENT_SOURCE_DIR}/testInversion.cxx+)

ROOTTEST_ADD_TEST(testKalman
                MACRO ${CMAKE_CURRENT_SOURCE_DIR}/testKalman.cxx+)

ROOTTEST_ADD_TEST(testOperations
                MACRO ${CMAKE_CURRENT_SOURCE_DIR}/testOperations.cxx+)

ROOTTEST_ADD_TEST(testSMatrix
                MACRO ${CMAKE_CURRENT_SOURCE_DIR}/testSMatrix.cxx+)
