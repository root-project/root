if(ROOT_unuran_FOUND)

ROOTTEST_ADD_TEST(roottest-root-math-unuran-unuranDiscrete
                  MACRO ${ROOT_SOURCE_DIR}/math/unuran/test/unuranDiscrete.cxx+)

ROOTTEST_ADD_TEST(roottest-root-math-unuran-unuranDistr
                  MACRO ${ROOT_SOURCE_DIR}/math/unuran/test/unuranDistr.cxx+)

ROOTTEST_ADD_TEST(roottest-root-math-unuran-unuranMulti2D
                  MACRO ${ROOT_SOURCE_DIR}/math/unuran/test/unuranMulti2D.cxx+)

ROOTTEST_ADD_TEST(roottest-root-math-unuran-unuranMultiDim
                  MACRO ${ROOT_SOURCE_DIR}/math/unuran/test/unuranMultiDim.cxx+)

ROOTTEST_ADD_TEST(roottest-root-math-unuran-unuranSimple
                  MACRO ${ROOT_SOURCE_DIR}/math/unuran/test/unuranSimple.cxx+)

endif()
