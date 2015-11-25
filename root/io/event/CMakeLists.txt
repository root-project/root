#-------------------------------------------------------------------------------
#
# Placeholder file to translate the tests to the new CTest system. Meanwhile we
# define a CTest test that calls 'make' in ${CMAKE_CURRENT_SOURCE_DIR}
#
#-------------------------------------------------------------------------------
ROOTTEST_ADD_TEST(event
                  COMMAND make
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  DEPENDS roottest-scripts-utils)
