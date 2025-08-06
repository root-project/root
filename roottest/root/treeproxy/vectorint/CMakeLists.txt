ROOTTEST_ADD_TEST(copymacros
                  COMMAND ${CMAKE_COMMAND} -E echo "copy files"
                  COPY_TO_BUILDDIR dude.h dude.C dude.cxx runvectorint.C runvectorintEmu.C
                  FIXTURES_SETUP root-treeproxy-vectorint-copymacros-fixture)

ROOTTEST_COMPILE_MACRO(runvectorint.C
                       FIXTURES_REQUIRED root-treeproxy-vectorint-copymacros-fixture
                       FIXTURES_SETUP root-treeproxy-vectorint-runvectorint_C-fixture)

ROOTTEST_ADD_TEST(createfile
                  MACRO runvectorint.C+
                  MACROARG 3
                  FIXTURES_REQUIRED root-treeproxy-vectorint-copymacros-fixture
                                    root-treeproxy-vectorint-runvectorint_C-fixture
                  FIXTURES_SETUP root-treeproxy-vectorint-createfile-fixture)

ROOTTEST_ADD_TEST(createselector
                  MACRO runvectorint.C+
                  MACROARG 4
                  FIXTURES_REQUIRED root-treeproxy-vectorint-copymacros-fixture
                                    root-treeproxy-vectorint-runvectorint_C-fixture
                                    root-treeproxy-vectorint-createfile-fixture
                  FIXTURES_SETUP root-treeproxy-vectorint-createselector-fixture)

ROOTTEST_ADD_TEST(vectorint
                  MACRO runvectorint.C+
                  MACROARG 5
                  OUTREF vectorint.ref
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  FIXTURES_REQUIRED root-treeproxy-vectorint-copymacros-fixture
                                    root-treeproxy-vectorint-runvectorint_C-fixture
                                    root-treeproxy-vectorint-createfile-fixture
                                    root-treeproxy-vectorint-createselector-fixture)

ROOTTEST_COMPILE_MACRO(runvectorintEmu.C
                       FIXTURES_REQUIRED root-treeproxy-vectorint-copymacros-fixture
                       FIXTURES_SETUP root-treeproxy-vectorint-runvectorintEmu_C-fixture)

ROOTTEST_ADD_TEST(createselectorEmu
                  MACRO runvectorintEmu.C+
                  MACROARG 4
                  FIXTURES_REQUIRED root-treeproxy-vectorint-copymacros-fixture
                                    root-treeproxy-vectorint-runvectorintEmu_C-fixture
                                    root-treeproxy-vectorint-createfile-fixture
                  FIXTURES_SETUP root-treeproxy-vectorint-createselectorEmu-fixture)

ROOTTEST_ADD_TEST(vectorintEmu
                  MACRO runvectorintEmu.C+
                  MACROARG 5
                  OUTREF vectorintEmu.ref
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  FIXTURES_REQUIRED root-treeproxy-vectorint-copymacros-fixture
                                    root-treeproxy-vectorint-runvectorintEmu_C-fixture
                                    root-treeproxy-vectorint-createfile-fixture
                                    root-treeproxy-vectorint-createselectorEmu-fixture)
