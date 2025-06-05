ROOTTEST_ADD_TEST(write
  MACRO execWriteCustomCollection.cxx+
  OUTREF execWriteCustomCollection.ref
  FIXTURES_SETUP customcollection-write)

ROOTTEST_ADD_TEST(read
  MACRO runcustomCollection.C
  OUTREF customCollection.ref
  FIXTURES_REQUIRED customcollection-write)