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

# Set global defines.
add_definitions(
#  -DClingWorkAroundJITandInline
#  -DClingWorkAroundNoPrivateClassIO 
#  -DClingWorkAroundTClassUpdateDouble32 
#  -DClingWorkAroundMissingDynamicScope 
#  -DClingWorkAroundUnnamedInclude 
#  -DClingWorkAroundStripDefaultArg 
#  -DClingWorkAroundPrintfIssues 
#  -DClingWorkAroundLackOfModule 
#  -DClingWorkAroundProxyConfusion 
#  -DClingWorkAroundScriptClassDef 
#  -DClingWorkAroundMultipleInclude 
#  -DClingWorkAroundExtraParensWithImplicitAuto 
#  -DClingWorkAroundNoPrivateClassIO 
#  -DClingWorkAroundBrokenRecovery 
#  -DClingWorkAroundBrokenUnnamedReturn 
#  -DClingWorkAroundUnnamedDetection 
#  -DClingWorkAroundUnnamedInclude 
#  -DClingWorkAroundJITfullSymbolResolution 
#  -DClingWorkAroundDeletedSourceFile 
#  -DClingWorkAroundValuePrinterNotFullyQualified 
#  -DClingWorkAroundNoDotNamespace 
#  -DClingWorkAroundNoDotInclude 
#  -DClingWorkAroundNoDotOptimization 
#  -DClingWorkAroundUnnamedIncorrectFileLoc 
#  -DClingWorkAroundUnloadingIOSTREAM 
#  -DClingWorkAroundUnloadingVTABLES 
)
