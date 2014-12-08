#---Custom CTest settings---------------------------------------------------

set(CTEST_CUSTOM_TESTS_IGNORE ${CTEST_CUSTOM_TESTS_IGNORE}
    test-stressgui
    test-stressproof)

if(WIN32)
  set(CTEST_CUSTOM_TESTS_IGNORE ${CTEST_CUSTOM_TESTS_IGNORE} 
      test-tcollex)
endif()

if(CTEST_BUILD_NAME MATCHES icc14)  #  sse tests of vc fail for icc compiler
  set(CTEST_CUSTOM_TESTS_IGNORE ${CTEST_CUSTOM_TESTS_IGNORE}
      vc-math_sse
      vc-math_VC_LOG_ILP2_sse
      vc-math_VC_LOG_ILP_sse)
endif()


if(CTEST_BUILD_NAME MATCHES mac107)  #  sse tests of vc fail for mac107 compiler
  set(CTEST_CUSTOM_TESTS_IGNORE ${CTEST_CUSTOM_TESTS_IGNORE}
      vc-gather_VC_USE_BSF_GATHERS_sse
      vc-gather_VC_USE_SET_GATHERS_sse
      vc-load_sse
      vc-math_sse
      vc-math_VC_LOG_ILP2_sse
      vc-math_VC_LOG_ILP_sse
      vc-scalaraccess_sse
      vc-swizzles_sse
      vc-utils_sse)
endif()

if(CTEST_BUILD_NAME MATCHES vc9)  #  some tests fail for vc9 compiler
  set(CTEST_CUSTOM_TESTS_IGNORE ${CTEST_CUSTOM_TESTS_IGNORE}
      mathcore-binarySearchTime
      mathcore-testkdTreeBinning
      test-stresstmva
      tutorial-math-kdTreeBinning
      tutorial-roostats-HybridOriginalDemo
      tutorial-roostats-rs801_HypoTestInverterOriginal)
endif()


