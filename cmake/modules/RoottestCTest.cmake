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

# Cling workaround support.

# Set CLING_VERSION analog to the existing make build system.
set(CLING_VERSION 5)

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

# Variables to be used in CMakeLists.txt files.

set(ClingWorkAroundMissingDynamicScope              TRUE)
set(ClingWorkAroundUnloadingIOSTREAM                TRUE)
set(ClingWorkAroundUnnamedInclude                   TRUE)
set(ClingWorkAroundMissingSmartInclude              TRUE)
set(ClingWorkAroundNoDotInclude                     TRUE)
set(ClingWorkAroundMissingAutoLoadingForTemplates   TRUE)
set(ClingWorkAroundAutoParseUsingNamespace          TRUE)
set(ClingWorkAroundTClassUpdateDouble32             TRUE)
set(ClingWorkAroundAutoParseTooPrecise              TRUE)
set(ClingWorkAroundAutoParseDeclaration             TRUE)
set(ClingWorkAroundMissingUnloading                 TRUE)
set(ClingWorkAroundJITandInline                     TRUE)
set(ClingWorkAroundBrokenUnnamedReturn              TRUE)
