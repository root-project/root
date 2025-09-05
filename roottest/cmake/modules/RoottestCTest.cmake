#-------------------------------------------------------------------------------
#
#  RootCTest.cmake
#
#  Setup file for CTest.
#
#-------------------------------------------------------------------------------

# Set the CTest build name.
set(CTEST_BUILD_NAME ${ROOT_ARCHITECTURE}-${CMAKE_BUILD_TYPE})
#message("-- Set CTest build name to: ${CTEST_BUILD_NAME}")

# Enable the creation and submission of dashboards.
include(CTest)

# Enable testing for CTest.
enable_testing()

# Copy the CTestCustom.cmake file into the build directory.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

# Cling workaround support.

# Set CINT_VERSION analog to the existing make build system.
set(CINT_VERSION cling)

# Cling workaround defines.
# Set of macros to avoid using features not yet implemented by cling.

add_definitions(
  -DClingWorkAroundMissingDynamicScope
  -DClingWorkAroundUnnamedInclude
  -DClingWorkAroundMissingSmartInclude
  -DClingWorkAroundNoDotInclude
  -DClingWorkAroundMissingAutoLoadingForTemplates
  -DClingWorkAroundAutoParseUsingNamespace
  -DClingWorkAroundTClassUpdateDouble32
  -DClingWorkAroundAutoParseDeclaration
  -DClingWorkAroundMissingUnloading
  -DClingWorkAroundBrokenUnnamedReturn
  -DClingWorkAroundUnnamedDetection2
  -DClingWorkAroundNoPrivateClassIO
  -DClingWorkAroundUnloadingVTABLES
)

# Variables to be used in CMakeLists.txt files.

set(ClingWorkAroundMissingDynamicScope              TRUE)
set(ClingWorkAroundUnnamedInclude                   TRUE)      # See https://sft.its.cern.ch/jira/browse/ROOT-4763
set(ClingWorkAroundMissingSmartInclude              TRUE)      # disabled in Makefile-based?
set(ClingWorkAroundNoDotInclude                     TRUE)      # See trello card about .include
set(ClingWorkAroundMissingAutoLoadingForTemplates   TRUE)      # See: https://sft.its.cern.ch/jira/browse/ROOT-4786
set(ClingWorkAroundAutoParseUsingNamespace          TRUE)      # See https://sft.its.cern.ch/jira/browse/ROOT-6317
set(ClingWorkAroundTClassUpdateDouble32             TRUE)      # See https://sft.its.cern.ch/jira/browse/ROOT-5857
set(ClingWorkAroundAutoParseDeclaration             TRUE)      # See https://sft.its.cern.ch/jira/browse/ROOT-6320
set(ClingWorkAroundMissingUnloading                 TRUE)      # disabled in Makefile-based?
set(ClingWorkAroundBrokenUnnamedReturn              TRUE)      # See https://sft.its.cern.ch/jira/browse/ROOT-4719
set(ClingWorkAroundNoPrivateClassIO                 TRUE)      # See https://sft.its.cern.ch/jira/browse/ROOT-4865
set(ClingWorkAroundUnnamedDetection2                TRUE)      # See https://sft.its.cern.ch/jira/browse/ROOT-8025
set(ClingWorkAroundUnloadingVTABLES                 TRUE)      # See https://sft.its.cern.ch/jira/browse/ROOT-6219

set(PYROOT_EXTRAFLAGS --fixcling)
