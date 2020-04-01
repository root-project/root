if(ROOT_unuran_FOUND)
        ROOTTEST_ADD_TEST(unuranDiscrete
                MACRO ${CMAKE_CURRENT_SOURCE_DIR}/unuranDiscrete.cxx+
                LABELS longtest)

        ROOTTEST_ADD_TEST(unuranDistr
                MACRO ${CMAKE_CURRENT_SOURCE_DIR}/unuranDistr.cxx+)

        ROOTTEST_ADD_TEST(unuranMulti2D
                MACRO ${CMAKE_CURRENT_SOURCE_DIR}/unuranMulti2D.cxx+)

        ROOTTEST_ADD_TEST(unuranMultiDim
                MACRO ${CMAKE_CURRENT_SOURCE_DIR}/unuranMultiDim.cxx+
                LABELS longtest)

        ROOTTEST_ADD_TEST(unuranSimple
                MACRO ${CMAKE_CURRENT_SOURCE_DIR}/unuranSimple.cxx+)
endif()