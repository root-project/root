ROOTTEST_ADD_TEST(evolution
                  MACRO evolution.C+
                  FIXTURES_SETUP root-meta-alignment-evolution-fixture)

ROOTTEST_ADD_TEST(emulation
                  MACRO emulation.C+
                  FIXTURES_REQUIRED root-meta-alignment-evolution-fixture)

ROOTTEST_ADD_TEST(vecalloc
                  MACRO vecalloc.C+)
