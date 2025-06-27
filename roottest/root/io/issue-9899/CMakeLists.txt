ROOTTEST_ADD_TEST(write_cmssw_class_v2
                  MACRO write_cmssw_class_v2.cxx+
                  OUTREF write_cmssw_class_v2.ref
                  FIXTURES_SETUP root-io-issue-9899-write_cmssw_class_v2-fixture)

ROOTTEST_ADD_TEST(read_cmssw_class_v3
                  MACRO read_cmssw_class_v3.cxx+
                  OUTREF read_cmssw_class_v3.ref
                  FIXTURES_REQUIRED root-io-issue-9899-write_cmssw_class_v2-fixture)

