ROOTTEST_ADD_TEST(write
                  MACRO writeGrowingPairVector.C+
                  FIXTURES_SETUP root-io-emulatedGrow-write-fixture)

ROOTTEST_ADD_TEST(read
                  MACRO readGrowingPairVector.C
                  FIXTURES_REQUIRED root-io-emulatedGrow-write-fixture)
