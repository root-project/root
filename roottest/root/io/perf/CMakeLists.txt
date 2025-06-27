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

ROOTTEST_ADD_TESTDIRS()