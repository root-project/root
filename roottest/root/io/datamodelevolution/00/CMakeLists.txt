ROOTTEST_ADD_TEST(test1
                  MACRO test1.cxx+
                  OUTREF test1.ref
                  LABELS longtest
                  FIXTURES_SETUP root-io-datamodelevolution-00-test1-fixture)

ROOTTEST_ADD_TEST(test2
                  MACRO test2.cxx+
                  OUTREF test2.ref
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-datamodelevolution-00-test1-fixture
                  FIXTURES_SETUP root-io-datamodelevolution-00-test2-fixture)

ROOTTEST_ADD_TEST(test3
                  MACRO test3.cxx+
                  OUTREF test3.ref
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-datamodelevolution-00-test2-fixture)
