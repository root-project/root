#---Custom CTest settings---------------------------------------------------

set(CTEST_CUSTOM_TESTS_IGNORE ${CTEST_CUSTOM_TESTS_IGNORE} 
    test-stressgui)

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

