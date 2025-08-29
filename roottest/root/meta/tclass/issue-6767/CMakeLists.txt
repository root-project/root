ROOTTEST_GENERATE_DICTIONARY(libRepro
                             repro.h
                             LINKDEF reproLinkDef.h
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-meta-tclass-6767-dict-fixture)

ROOTTEST_ADD_TEST(reproducer
                  MACRO repro.C
                  OUTREF repro.ref
                  FIXTURES_REQUIRED root-meta-tclass-6767-dict-fixture)
