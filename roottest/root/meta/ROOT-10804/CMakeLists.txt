ROOTTEST_GENERATE_REFLEX_DICTIONARY(ROOT10804
                                    ROOT10804.h
                                    SELECTION selection.xml
                                    FIXTURES_SETUP root-meta-10804-fixture)

ROOTTEST_ADD_TEST(execROOT10804
                  MACRO execROOT10804.C
                  OUTREF execROOT10804.oref
                  ERRREF execROOT10804.eref
                  FIXTURES_REQUIRED root-meta-10804-fixture)
