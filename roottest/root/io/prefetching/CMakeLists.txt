#-------------------------------------------------------------------------------
#
# Placeholder file to translate the tests to the new CTest system. Meanwhile we
# define a CTest test that calls 'make' in ${CMAKE_CURRENT_SOURCE_DIR}
#
#-------------------------------------------------------------------------------
if(xrootd)
  ROOTTEST_ADD_OLDTEST(LABELS longtest)
endif()