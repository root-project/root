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

set(PYROOT_EXTRAFLAGS --fixcling)

# set ROOTTEST_OS_ID
if(APPLE)
  execute_process(COMMAND sw_vers "-productVersion"
                  COMMAND cut -d . -f 1-2
                  OUTPUT_VARIABLE osvers OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(ROOTTEST_OS_ID MacOSX)                
  set(ROOTTEST_OS_VERSION ${osvers})
elseif(WIN32)
  set(ROOTTEST_OS_ID Windows)
else()
  execute_process(COMMAND lsb_release -is OUTPUT_VARIABLE osid OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND lsb_release -rs OUTPUT_VARIABLE osvers OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(osid MATCHES Ubuntu)
    string(REGEX REPLACE "([0-9]+)[.].*" "\\1" osvers "${osvers}")
    set(ROOTTEST_OS_ID Ubuntu)                
    set(ROOTTEST_OS_VERSION ${osvers}) 
  elseif(osid MATCHES Scientific)
    string(REGEX REPLACE "([0-9]+)[.].*" "\\1" osvers "${osvers}")
    set(ROOTTEST_OS_ID Scientific)                
    set(ROOTTEST_OS_VERSION ${osvers}) 
  elseif(osid MATCHES Fedora)
    string(REGEX REPLACE "([0-9]+)" "\\1" osvers "${osvers}")
    set(ROOTTEST_OS_ID Fedora)                
    set(ROOTTEST_OS_VERSION ${osvers}) 
  elseif(osid MATCHES CentOS)
    string(REGEX REPLACE "([0-9]+)[.].*" "\\1" osvers "${osvers}")
    set(ROOTTEST_OS_ID CentOS)                
    set(ROOTTEST_OS_VERSION ${osvers}) 
  else()
    string(REGEX REPLACE "([0-9]+)[.].*" "\\1" osvers "${osvers}")
    set(ROOTTEST_OS_ID ${osid})                
    set(ROOTTEST_OS_VERSION ${osvers}) 
  endif()
endif()

