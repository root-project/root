ROOTTEST_ADD_TEST(copylogon
                  COMMAND ${CMAKE_COMMAND} -E echo "copy rootlogon.C file"
                  COPY_TO_BUILDDIR rootlogon.C
                  FIXTURES_SETUP root-tree-stl-copylogon-fixture)

ROOTTEST_COMPILE_MACRO(MyClass.h
                       FIXTURES_SETUP root-tree-stl-MyClass-fixture)

ROOTTEST_ADD_TEST(mixingInterp
                  MACRO runmixing.C
                  OUTREF mixingInterp.ref
                  FIXTURES_REQUIRED root-tree-stl-MyClass-fixture root-tree-stl-copylogon-fixture)

ROOTTEST_COMPILE_MACRO(runmixing.C
                       FIXTURES_SETUP root-tree-stl-runmixing-fixture)

ROOTTEST_ADD_TEST(mixingCompiled
                  MACRO runmixing.C+
                  OUTREF mixingCompiled.ref
                  FIXTURES_REQUIRED root-tree-stl-runmixing-fixture root-tree-stl-copylogon-fixture)

ROOTTEST_COMPILE_MACRO(runmixingGood.C
                       FIXTURES_SETUP root-tree-stl-runmixingGood-fixture)

ROOTTEST_ADD_TEST(mixingGood
                  MACRO runmixingGood.C+
                  OUTREF mixingGood.ref
                  FIXTURES_REQUIRED root-tree-stl-runmixingGood-fixture root-tree-stl-copylogon-fixture)

ROOTTEST_ADD_TEST(VecPtr
                  MACRO execVecPtr.cxx+
                  OUTREF execVecPtr.ref
                  FIXTURES_REQUIRED root-tree-stl-copylogon-fixture)

ROOTTEST_ADD_TEST(vstr
                  MACRO runvstr.C+
                  OUTREF vstr.ref
                  FIXTURES_REQUIRED root-tree-stl-copylogon-fixture)

ROOTTEST_COMPILE_MACRO(runsimple.C
                       FIXTURES_SETUP root-tree-stl-runsimple-fixture)

ROOTTEST_ADD_TEST(simple
                  MACRO runsimple.C+
                  OUTREF simple.ref
                  FIXTURES_REQUIRED root-tree-stl-copylogon-fixture root-tree-stl-runsimple-fixture)

