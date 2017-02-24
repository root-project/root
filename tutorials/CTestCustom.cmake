#---Custom CTest settings---------------------------------------------------

set(CTEST_CUSTOM_TESTS_IGNORE ${CTEST_CUSTOM_TESTS_IGNORE} tutorial-pyroot-zdemo)

if("$ENV{COMPILER}" STREQUAL "classic") #  TTreeProcessorM{T,P} are not available
  list(APPEND CTEST_CUSTOM_TESTS_IGNORE tutorial-multicore-mp102_readNtuplesFillHistosAndFit)
endif()


