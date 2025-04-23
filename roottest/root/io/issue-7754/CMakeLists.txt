ROOTTEST_ADD_TEST(write_cmssw_class_v2
                  MACRO write_cmssw_class_v2.cxx+
                  OUTREF write_cmssw_class_v2.ref)

ROOTTEST_ADD_TEST(read_cmssw_class_v3
                  MACRO read_cmssw_class_v3.cxx+
                  OUTREF read_cmssw_class_v3.ref
                  DEPENDS write_cmssw_class_v2 )

set(RootExternalIncludes -e "gSystem->AddIncludePath(\"-I${CMAKE_CURRENT_SOURCE_DIR}/mwe/include\");")

ROOTTEST_ADD_TEST(read_mwe_0
                  MACRO mwe/combined.cxx+
                  MACROARG 0
                  OUTREF mwe/read_mwe_0.ref
                  COPY_TO_BUILDDIR mwe/pluto_oldpers.root)

ROOTTEST_ADD_TEST(read_mwe_1
                  MACRO mwe/combined.cxx+
                  MACROARG 1
                  OUTREF mwe/read_mwe_1.ref
                  COPY_TO_BUILDDIR mwe/pluto_newpers.root
                  DEPENDS read_mwe_0) # To avoid both test building the script at the same time.

