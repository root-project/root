ROOTTEST_ADD_TEST(test1
                  MACRO test1.cxx+
                  OUTREF test1.ref
                  LABELS longtest
                  FIXTURES_SETUP io-datamodelevolution-00-test1)

ROOTTEST_ADD_TEST(test2
                  MACRO test2.cxx+
                  OUTREF test2.ref
                  LABELS longtest
                  FIXTURES_REQUIRED io-datamodelevolution-00-test1
                  FIXTURES_SETUP io-datamodelevolution-00-test2)

ROOTTEST_ADD_TEST(test3
                  MACRO test3.cxx+
                  OUTREF test3.ref
                  LABELS longtest
                  FIXTURES_REQUIRED io-datamodelevolution-00-test2)
