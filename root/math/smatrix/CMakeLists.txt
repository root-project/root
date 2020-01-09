if(ROOT_SOURCE_DIR)
    ROOTTEST_ADD_TEST(testInversion
                MACRO ${ROOT_SOURCE_DIR}/math/smatrix/test/testInversion.cxx+)

    ROOTTEST_ADD_TEST(testKalman
                MACRO ${ROOT_SOURCE_DIR}/math/smatrix/test/testKalman.cxx+)

    ROOTTEST_ADD_TEST(testOperations
                MACRO ${ROOT_SOURCE_DIR}/math/smatrix/test/testOperations.cxx+)

    ROOTTEST_ADD_TEST(testSMatrix
                MACRO ${ROOT_SOURCE_DIR}/math/smatrix/test/testSMatrix.cxx+)
elseif(ROOTSYS)
    ROOTTEST_ADD_TEST(testInversion
                MACRO ${ROOTSYS}/math/smatrix/test/testInversion.cxx+)

    ROOTTEST_ADD_TEST(testKalman
                MACRO ${ROOTSYS}/math/smatrix/test/testKalman.cxx+)

    ROOTTEST_ADD_TEST(testOperations
                MACRO ${ROOTSYS}/math/smatrix/test/testOperations.cxx+)

    ROOTTEST_ADD_TEST(testSMatrix
                MACRO ${ROOTSYS}/math/smatrix/test/testSMatrix.cxx+)
endif()
