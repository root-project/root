ROOTTEST_COMPILE_MACRO(Toplevel.C
                       FIXTURES_SETUP root-io-interface-Toplevel-fixture)

ROOTTEST_COMPILE_MACRO(ToplevelClass.C
                       FIXTURES_SETUP root-io-interface-ToplevelClass-fixture)

ROOTTEST_ADD_TEST(Toplevel
                  MACRO runToplevel.C
                  FIXTURES_REQUIRED root-io-interface-Toplevel-fixture root-io-interface-ToplevelClass-fixture
                  OUTREF Toplevel.ref
                  FIXTURES_SETUP root-io-interface-Toplevel-run-fixture)

ROOTTEST_ADD_TEST(currentFileKey
                  MACRO runcurrentFileKey.C
                  FIXTURES_REQUIRED root-io-interface-Toplevel-run-fixture
                  OUTREF currentFileKey.ref)
