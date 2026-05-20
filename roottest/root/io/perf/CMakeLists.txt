ROOTTEST_ADD_TEST(julius
                  MACRO julius.C+)

ROOTTEST_COMPILE_MACRO(WriteBuffer.C
                       FIXTURES_SETUP root-io-perf-WriteBuffer-fixture)

ROOTTEST_COMPILE_MACRO(ReadBuffer.C
                       FIXTURES_SETUP root-io-perf-ReadBuffer-fixture)

foreach(k 1 2)
  ROOTTEST_ADD_TEST(WriteBuffer-${k}
                    MACRO WriteBuffer.C+
                    MACROARG "${k},10000000"
                    FIXTURES_REQUIRED root-io-perf-WriteBuffer-fixture)
  ROOTTEST_ADD_TEST(ReadBuffer-${k}
                    MACRO ReadBuffer.C+
                    MACROARG "${k},10000000"
                    FIXTURES_REQUIRED root-io-perf-ReadBuffer-fixture)
endforeach()

ROOTTEST_ADD_TEST(directories
                  MACRO directories.C+)

# these are very long tests, writing huge files,
# not run them by default
if(very_long_tests)
  ROOTTEST_ADD_TEST(ShortTest
                    MACRO SergeiShortTest.C+
                    LABELS longtest)

  ROOTTEST_ADD_TEST(HardTest
                    MACRO SergeiHardTest.C+
                    MACROARG 1
                    LABELS longtest)
endif()

ROOTTEST_ADD_TESTDIRS()
