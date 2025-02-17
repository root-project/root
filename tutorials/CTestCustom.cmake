#---Custom CTest settings---------------------------------------------------

set(CTEST_CUSTOM_TESTS_IGNORE ${CTEST_CUSTOM_TESTS_IGNORE} tutorial-pyroot-zdemo)

if("$ENV{COMPILER}" STREQUAL "classic") 
  #  TTreeProcessorM{T,P} are not available
  list(APPEND CTEST_CUSTOM_TESTS_IGNORE tutorial-multicore-mp102_readNtuplesFillHistosAndFit)
  #  pthread is not retained on ubuntus when building root.exe
  list(APPEND CTEST_CUSTOM_TESTS_IGNORE
       tutorial-multicore-mt101_fillNtuples
       tutorial-multicore-mt001_fillHistos
       tutorial-multicore-mt201_parallelHistoFill)
endif()

if (CTEST_BUILD_NAME MATCHES aarch64 AND CTEST_BUILD_NAME MATCHES dbg)
  # these tutorials are disabled as they timeout
  list(APPEND CTEST_CUSTOM_TESTS_IGNORE
       tutorial-roofit-roostats-StandardBayesianNumericalDemo
       tutorial-roofit-roostats-OneSidedFrequentistUpperLimitWithBands
       tutorial-machine_learning-TMVAClassification
       tutorial-machine_learning-TMVARegression
       tutorial-machine_learning-TMVAMulticlass
       tutorial-machine_learning-TMVAMulticlassApplication
       tutorial-machine_learning-TMVARegressionApplication
       tutorial-machine_learning-TMVAClassificationApplication
       tutorial-roofit-roostats-TwoSidedFrequentistUpperLimitWithBands)
endif()

if (CTEST_BUILD_NAME MATCHES aarch64)
  # The new triangulation does not work on ARM
  list(APPEND CTEST_CUSTOM_TESTS_IGNORE tutorial-mlp-mlpRegression)
endif()
