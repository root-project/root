#-------------------------------------------------------------------------------
#
#  RootCTest.cmake
#
#  Setup file for CTest.
#
#-------------------------------------------------------------------------------

# Set the CTest build name.
set(CTEST_BUILD_NAME ${ROOT_ARCHITECTURE}-${CMAKE_BUILD_TYPE})
message("-- Set CTest build name to: ${CTEST_BUILD_NAME}")

# Enable the creation and submission of dashboards.
include(CTest)

# Enable testing for CTest.
enable_testing()

# Copy the CTestCustom.cmake file into the build directory.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake ${CMAKE_BINARY_DIR} COPYONLY)

# Cling workaround defines.
# Set of macros to avoid using features not yet implemented by cling.

add_definitions(
  -DClingWorkAroundMissingDynamicScope 
  -DClingWorkAroundUnloadingIOSTREAM
  -DClingWorkAroundUnnamedInclude
  -DClingWorkAroundMissingSmartInclude
  -DClingWorkAroundNoDotInclude
  -DClingWorkAroundMissingAutoLoadingForTemplates
  -DClingWorkAroundAutoParseUsingNamespace
  -DClingWorkAroundTClassUpdateDouble32
  -DClingWorkAroundAutoParseTooPrecise
  -DClingWorkAroundAutoParseDeclaration  
  -DClingWorkAroundMissingUnloading
  -DClingWorkAroundJITandInline
  -DClingWorkAroundBrokenUnnamedReturn
)
