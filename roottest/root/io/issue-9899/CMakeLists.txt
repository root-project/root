ROOTTEST_ADD_TEST(write_cmssw_class_v2
                  MACRO write_cmssw_class_v2.cxx+
                  OUTREF write_cmssw_class_v2.ref)

ROOTTEST_ADD_TEST(read_cmssw_class_v3
                  MACRO read_cmssw_class_v3.cxx+
                  OUTREF read_cmssw_class_v3.ref
                  DEPENDS write_cmssw_class_v2 )

