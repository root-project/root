#---Custom CTest settings---------------------------------------------------

if (CTEST_BUILD_NAME MATCHES aarch64 AND CTEST_BUILD_NAME MATCHES dbg)
  # these tests are disabled as they timeout
  list(APPEND CTEST_CUSTOM_TESTS_IGNORE mathcore-testMathRandom)
endif()
