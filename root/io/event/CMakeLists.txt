#-------------------------------------------------------------------------------
#
# Placeholder file to translate the tests to the new CTest system. Meanwhile we
# define a CTest test that calls 'make' in ${CMAKE_CURRENT_SOURCE_DIR}
#
#-------------------------------------------------------------------------------
if(NOT MSVC OR win_broken_tests)
    ROOTTEST_ADD_TEST(roottest-root-io-event
                      COMMAND ${ROOT_GMAKE_PROGRAM} cleantest
                      WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                      DEPENDS roottest-scripts-utils)
endif()
