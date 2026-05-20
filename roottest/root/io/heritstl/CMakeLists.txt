ROOTTEST_COMPILE_MACRO(classes.C
                       FIXTURES_SETUP root-io-heritstl-classes-fixture)

ROOTTEST_COMPILE_MACRO(minostest.C
                       FIXTURES_SETUP root-io-heritstl-minostest-fixture)

ROOTTEST_COMPILE_MACRO(runbaseString.C
                       FIXTURES_SETUP root-io-heritstl-runbaseString-fixture)


ROOTTEST_ADD_TEST(minostest
                  MACRO runminostest.C
                  FIXTURES_REQUIRED root-io-heritstl-classes-fixture root-io-heritstl-minostest-fixture
                  OUTREF minostest.ref
                  OUTCNVCMD sed -e s,offset=[[:space:]]*[0-9]*,offset=na,g)

ROOTTEST_ADD_TEST(stress_root
                  MACRO runbaseString.C+
                  MACROARG 0
                  FIXTURES_REQUIRED root-io-heritstl-runbaseString-fixture
                  FIXTURES_SETUP root-io-heritstl-stress_root-fixture)

ROOTTEST_ADD_TEST(baseString
                  MACRO runbaseString.C+
                  OUTREF baseString.ref
                  FIXTURES_REQUIRED root-io-heritstl-stress_root-fixture root-io-heritstl-runbaseString-fixture)
