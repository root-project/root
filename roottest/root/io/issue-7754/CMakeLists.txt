ROOTTEST_ADD_TEST(write_cmssw_class_v2
                  MACRO write_cmssw_class_v2.cxx+
                  OUTREF write_cmssw_class_v2.ref
                  FIXTURES_SETUP root-io-issue-7754-write_cmssw_class_v2-fixture)

ROOTTEST_ADD_TEST(read_cmssw_class_v3
                  MACRO read_cmssw_class_v3.cxx+
                  OUTREF read_cmssw_class_v3.ref
                  FIXTURES_REQUIRED root-io-issue-7754-write_cmssw_class_v2-fixture)

ROOTTEST_ADD_TEST(read_mwe_0
                  MACRO mwe/combined.cxx+
                  MACROARG 0
                  ROOTEXE_OPTS -e "gSystem->AddIncludePath(\"-I${CMAKE_CURRENT_SOURCE_DIR}/mwe/include\")"
                  OUTREF mwe/read_mwe_0.ref
                  COPY_TO_BUILDDIR mwe/pluto_oldpers.root
                  FIXTURES_SETUP root-io-issue-7754-read_mwe_0-fixture)

ROOTTEST_ADD_TEST(read_mwe_1
                  MACRO mwe/combined.cxx+
                  MACROARG 1
                  ROOTEXE_OPTS -e "gSystem->AddIncludePath(\"-I${CMAKE_CURRENT_SOURCE_DIR}/mwe/include\")"
                  OUTREF mwe/read_mwe_1.ref
                  COPY_TO_BUILDDIR mwe/pluto_newpers.root
                  FIXTURES_REQUIRED root-io-issue-7754-read_mwe_0-fixture) # To avoid both test building the script at the same time

