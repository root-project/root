ROOTTEST_ADD_TEST(copyfiles
                  COMMAND ${CMAKE_COMMAND} -E echo "copy files"
                  COPY_TO_BUILDDIR Event1.root Event2.root Event3.root
                  FIXTURES_SETUP root-tree-selector-copyfiles-fixture)

ROOTTEST_ADD_TEST(selector
                  MACRO run.C
                  OUTREF selector.ref
                  OUTCNVCMD grep -v reloading
                  FIXTURES_REQUIRED root-tree-selector-copyfiles-fixture)

ROOTTEST_COMPILE_MACRO(sel01.C
                       FIXTURES_SETUP root-tree-selector-sel01-fixture)

ROOTTEST_ADD_TEST(SelectorCintTest
                  MACRO runSelectorCintTest.C
                  OUTREF SelectorCintTest.ref
                  OUTCNVCMD grep -v unloading
                  FIXTURES_REQUIRED root-tree-selector-sel01-fixture root-tree-selector-copyfiles-fixture)

ROOTTEST_COMPILE_MACRO(emptysel.C
                       FIXTURES_SETUP root-tree-selector-emptysel-fixture)

ROOTTEST_ADD_TEST(GetSelector
                  MACRO execGetSelector.C
                  OUTREF execGetSelector.ref
                  FIXTURES_REQUIRED root-tree-selector-emptysel-fixture)

ROOTTEST_ADD_TEST(LHEF
                  MACRO execLHEF.C
                  OUTREF execLHEF.ref
                  COPY_TO_BUILDDIR unweighted.root)

ROOTTEST_ADD_TEST(WithAbort
                  MACRO execWithAbort.C
                  OUTREF execWithAbort.ref
                  OUTCNVCMD grep -v reloading
                  FIXTURES_REQUIRED root-tree-selector-copyfiles-fixture)
