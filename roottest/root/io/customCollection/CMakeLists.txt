ROOTTEST_ADD_TEST(write
                  MACRO execWriteCustomCollection.cxx+
                  OUTREF execWriteCustomCollection.ref
                  FIXTURES_SETUP root-io-customCollection-write-fixture)

ROOTTEST_ADD_TEST(read
                  MACRO runcustomCollection.C
                  OUTREF customCollection.ref
                  FIXTURES_REQUIRED root-io-customCollection-write-fixture)