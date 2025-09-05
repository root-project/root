ROOTTEST_COMPILE_MACRO(Data.cxx
                       FIXTURES_SETUP root-treeformula-array-Data_cxx-fixture)

ROOTTEST_ADD_TEST(filltree
                  MACRO fillTree.cxx
                  ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L Data.cxx+\")"
                  FIXTURES_REQUIRED root-treeformula-array-Data_cxx-fixture
                  FIXTURES_SETUP root-treeformula-array-filltree-fixture)

ROOTTEST_ADD_TEST(arraytest
                  MACRO runArrayTest.C
                  ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L Data.cxx+\")"
                  OUTREF arrayDrawTest.good
                  FIXTURES_REQUIRED root-treeformula-array-Data_cxx-fixture
                                    root-treeformula-array-filltree-fixture)

ROOTTEST_ADD_TEST(eventlist
                  MACRO eventlist.C
                  COPY_TO_BUILDDIR ggss207.root
                  OUTREF arrayEventListTest.ref
                  FIXTURES_SETUP root-treeformula-array-eventlist-fixture)

ROOTTEST_ADD_TEST(nonSplit
                  MACRO nonsplit.C+
                  ERRREF nonsplit.erref)

# depend from eventlist because of file copying
ROOTTEST_ADD_TEST(boolOpt
                  MACRO boolOpt.C
                  OUTREF boolOptTest.ref
                  FIXTURES_REQUIRED root-treeformula-array-eventlist-fixture)

ROOTTEST_ADD_TEST(indexMultiDim
                  MACRO indexMultiDim.C
                  COPY_TO_BUILDDIR orange.root shorttrack.root
                  OUTREF indexMultiDim.ref)

ROOTTEST_COMPILE_MACRO(twodimvar.C
                       FIXTURES_SETUP root-treeformula-array-twodimvar-fixture)

ROOTTEST_ADD_TEST(twodimvar
                  MACRO runtwodimvar.C
                  OUTREF twodimvar.ref
                  FIXTURES_REQUIRED root-treeformula-array-twodimvar-fixture)

ROOTTEST_COMPILE_MACRO(NtpLib.C
                       FIXTURES_SETUP root-treeformula-array-NtpLib-fixture)

ROOTTEST_ADD_TEST(multidim
                  MACRO runmultidim.C
                  OUTREF multidim.ref
                  FIXTURES_REQUIRED root-treeformula-array-NtpLib-fixture)

ROOTTEST_COMPILE_MACRO(ScanString.C
                       FIXTURES_SETUP root-treeformula-array-ScanString-fixture)

ROOTTEST_ADD_TEST(ScanString
                  MACRO runScanString.C
                  OUTREF ScanString.ref
                  FIXTURES_REQUIRED root-treeformula-array-ScanString-fixture)

# compile macro before starting it to correcly load entries for dictionary
ROOTTEST_COMPILE_MACRO(runvectorOfvector.C
                       FIXTURES_SETUP root-treeformula-array-vectorOfvector-fixture)

ROOTTEST_ADD_TEST(vectorOfvector
                  MACRO runvectorOfvector.C+
                  OUTREF vectorOfvector.ref
                  FIXTURES_REQUIRED root-treeformula-array-vectorOfvector-fixture)

ROOTTEST_ADD_TEST(indexLoad
                  MACRO runindexLoad.C+
                  OUTREF indexLoad.ref)

ROOTTEST_ADD_TEST(vectorInVector
                  MACRO runvectorInVector.C
                  COPY_TO_BUILDDIR CaloTowers.root
                  OUTREF vectorInVector.ref)

ROOTTEST_ADD_TEST(nestedind
                  MACRO runnestedind.C
                  OUTREF nestedind.ref)

ROOTTEST_ADD_TEST(NestedFail
                  MACRO execNestedFail.C
                  OUTREF execNestedFail.ref)

ROOTTEST_ADD_TEST(OutOfBounds
                  MACRO execOutOfBounds.C
                  COPY_TO_BUILDDIR run133.root
                  OUTREF execOutOfBounds.ref)
