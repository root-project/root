ROOTTEST_COMPILE_MACRO(little.C
                       FIXTURES_SETUP root-io-fakeClass-little-fixture)

ROOTTEST_COMPILE_MACRO(little_v2.C
                       FIXTURES_SETUP root-io-fakeClass-little-v2-fixture)

ROOTTEST_ADD_TEST(printLittle
                  MACRO printLittle.C
                  OUTCNVCMD sed -e s,offset=[[:space:]]*[0-9]*,offset=na,g
                  OUTREF printLittle.ref
                  FIXTURES_REQUIRED root-io-fakeClass-little-fixture)

ROOTTEST_ADD_TEST(produceLittleFile
                  MACRO produceLittleFile.C
                  FIXTURES_REQUIRED root-io-fakeClass-little-fixture
                  FIXTURES_SETUP root-io-fakeClass-produceLittleFile-fixture)

ROOTTEST_ADD_TEST(forwardDeclAndFake
                  MACRO t01.C
                  OUTREF forwardDeclAndFake.ref
                  COPY_TO_BUILDDIR test.root
                  FIXTURES_REQUIRED root-io-fakeClass-little-fixture)

ROOTTEST_ADD_TEST(checkLittle0
                  MACRO checkLittle0.C
                  OUTREF checkLittle0.ref
                  # There is an unresolved library loading order problem
                  WILLFAIL
                  FIXTURES_REQUIRED root-io-fakeClass-produceLittleFile-fixture root-io-fakeClass-little-v2-fixture)

ROOTTEST_ADD_TEST(checkLittle
                  MACRO checkLittle.C
                  OUTREF checkLittle.ref
                  # There is an unresolved library loading order problem
                  WILLFAIL
                  FIXTURES_REQUIRED root-io-fakeClass-produceLittleFile-fixture root-io-fakeClass-little-v2-fixture)

ROOTTEST_ADD_TEST(checkLittle1
                  MACRO checkLittle1.C
                  OUTREF checkLittle1.ref
                  # There is an unresolved library loading order problem
                  WILLFAIL
                  FIXTURES_REQUIRED root-io-fakeClass-produceLittleFile-fixture root-io-fakeClass-little-v2-fixture)
