#-------------------------------------------------------------------------------
#
# Placeholder file to translate the tests to the new CTest system. Meanwhile we
# define a CTest test that calls 'make' in ${CMAKE_CURRENT_SOURCE_DIR}
#
#-------------------------------------------------------------------------------
# FIXME: Temporary workaround for runtime_cxxmodules;
if(NOT ROOT_runtime_cxxmodules_FOUND)
ROOTTEST_ADD_OLDTEST()
endif()
