if(ROOT_unuran_FOUND)
    if(ROOT_SOURCE_DIR)
        ROOTTEST_ADD_TEST(unuranDiscrete
                MACRO ${ROOT_SOURCE_DIR}/math/unuran/test/unuranDiscrete.cxx+
                LABELS longtest)

        ROOTTEST_ADD_TEST(unuranDistr
                MACRO ${ROOT_SOURCE_DIR}/math/unuran/test/unuranDistr.cxx+)

        ROOTTEST_ADD_TEST(unuranMulti2D
                MACRO ${ROOT_SOURCE_DIR}/math/unuran/test/unuranMulti2D.cxx+)

        ROOTTEST_ADD_TEST(unuranMultiDim
                MACRO ${ROOT_SOURCE_DIR}/math/unuran/test/unuranMultiDim.cxx+
                LABELS longtest)

        ROOTTEST_ADD_TEST(unuranSimple
                MACRO ${ROOT_SOURCE_DIR}/math/unuran/test/unuranSimple.cxx+)
    elseif(ROOTSYS)
        ROOTTEST_ADD_TEST(unuranDiscrete
                MACRO ${ROOTSYS}/math/unuran/test/unuranDiscrete.cxx+
                LABELS longtest)

        ROOTTEST_ADD_TEST(unuranDistr
                MACRO ${ROOTSYS}/math/unuran/test/unuranDistr.cxx+)

        ROOTTEST_ADD_TEST(unuranMulti2D
                MACRO ${ROOTSYS}/math/unuran/test/unuranMulti2D.cxx+)

        ROOTTEST_ADD_TEST(unuranMultiDim
                MACRO ${ROOTSYS}/math/unuran/test/unuranMultiDim.cxx+
                LABELS longtest)

        ROOTTEST_ADD_TEST(unuranSimple
                MACRO ${ROOTSYS}/math/unuran/test/unuranSimple.cxx+)
    endif()
endif()
