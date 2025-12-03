ROOTTEST_ADD_TEST(Draw_Problem
                  MACRO assert_Draw_Problem.C
                  COPY_TO_BUILDDIR Draw_Problem.root
                  ERRREF assert_Draw_Problem.ref)

ROOTTEST_ADD_TEST(SparseSelection
                  MACRO assertSparseSelection.C
                  COPY_TO_BUILDDIR badvector.root
                  ERRREF assertSparseSelection.ref)

ROOTTEST_ADD_TEST(ReferenceReturnType
                  MACRO execReferenceReturnType.C
                  OUTREF execReferenceReturnType.ref)

ROOTTEST_ADD_TEST(stlString
                  MACRO runstlString.C
                  COPY_TO_BUILDDIR EDM.root
                  OUTREF stlString.ref)

ROOTTEST_ADD_TEST(pair
                  MACRO runpair.C
                  COPY_TO_BUILDDIR RefTest.root
                  OUTREF pair.ref)

ROOTTEST_ADD_TEST(embedstl
                  MACRO runembedstl.C
                  COPY_TO_BUILDDIR Cumpb.root
                  OUTREF embedstl.ref)

ROOTTEST_COMPILE_MACRO(mapvector.C
                       FIXTURES_SETUP root-treeformula-stl-mapvector_C-fixture)

ROOTTEST_ADD_TEST(writemap
                  MACRO mapvector.C+
                  FIXTURES_REQUIRED root-treeformula-stl-mapvector_C-fixture
                  FIXTURES_SETUP root-treeformula-stl-writemap-fixture)

ROOTTEST_ADD_TEST(mapvector
                  MACRO runmapvector.C
                  OUTREF mapvector.ref
                  FIXTURES_REQUIRED root-treeformula-stl-writemap-fixture)

ROOTTEST_ADD_TEST(secondindex
                  MACRO runsecondindex.C
                  COPY_TO_BUILDDIR AthenaCrossSection.root
                  OUTREF secondindex.ref)
