ROOTTEST_ADD_TEST(selector
                  MACRO createtlorentzvecproxy.C
                  COPY_TO_BUILDDIR tlorentzvec.root analyze.h analyze.C cutting.h cutting.C
                  FIXTURES_SETUP root-treeproxy-stlvec-selector-fixture)

ROOTTEST_ADD_TEST(tlorentzvec
                  MACRO runtlorentzvec.C
                  OUTREF tlorentzvec.ref
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  FIXTURES_REQUIRED root-treeproxy-stlvec-selector-fixture)
