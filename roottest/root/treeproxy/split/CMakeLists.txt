ROOTTEST_ADD_TEST(NonSplitBranchMakeProxy
                  MACRO execNonSplitBranchMakeProxy.C
                  COPY_TO_BUILDDIR MyTree0_script.cxx MyTree99_script.cxx
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  OUTREF execNonSplitBranchMakeProxy.ref)
