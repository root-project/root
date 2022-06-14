#---Custom CTest settings---------------------------------------------------

set(CTEST_CUSTOM_TESTS_IGNORE ${CTEST_CUSTOM_TESTS_IGNORE} test-stressgui test-stressproof)

if(WIN32)
  set(CTEST_CUSTOM_TESTS_IGNORE ${CTEST_CUSTOM_TESTS_IGNORE} test-tcollex)
endif()

if(CTEST_BUILD_NAME MATCHES icc)  #  sse tests of vc fail for icc compiler
  set(CTEST_CUSTOM_TESTS_IGNORE ${CTEST_CUSTOM_TESTS_IGNORE}
      vc-math_sse
      vc-math_VC_LOG_ILP2_sse
      vc-math_VC_LOG_ILP_sse
      stressVdt)
elseif(CTEST_BUILD_NAME MATCHES clang7)
  set(CTEST_CUSTOM_TESTS_IGNORE ${CTEST_CUSTOM_TESTS_IGNORE}
      vc-swizzles_avx)
elseif (CTEST_BUILD_NAME MATCHES aarch64 AND CTEST_BUILD_NAME MATCHES dbg)
  # these tests are disabled as they timeout
  list(APPEND CTEST_CUSTOM_TESTS_IGNORE test-stressroostats test-stresstmva test-stressroostats-interpreted test-stresshistogram-interpreted test-stresshistogram test-stressgeometry test-tcollbm test-bench test-stressgeometry-interpreted)
endif()
