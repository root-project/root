if(NOT MSVC OR win_broken_tests)
    ROOTTEST_ADD_TEST(utils
                      COMMAND ${ROOT_GMAKE_PROGRAM} utils
                      WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                      FIXTURES_SETUP UtilsLibraryBuild)
endif()
