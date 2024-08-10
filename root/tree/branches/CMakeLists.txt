#-------------------------------------------------------------------------------
#
# Placeholder file to translate the tests to the new CTest system. Meanwhile we
# define a CTest test that calls 'make' in ${CMAKE_CURRENT_SOURCE_DIR}
#
#-------------------------------------------------------------------------------
ROOTTEST_ADD_OLDTEST(LABELS longtest)

# New tests follow

# ROOT-5983
ROOTTEST_ADD_TEST(initOffset
                  MACRO  initOffset.C+)