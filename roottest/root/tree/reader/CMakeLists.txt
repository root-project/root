# ROOT-10046
ROOTTEST_ADD_TEST(oddName
                  COPY_TO_BUILDDIR root-10046.root
                  MACRO oddName.C)

# ROOT-9757
ROOTTEST_ADD_TEST(numberBranchesRead
                  MACRO  test_numberBranchesRead.C
                  MACROARG "\"${CMAKE_CURRENT_SOURCE_DIR}/../../dataframe/Slimmed_TotemNTuple_9883.040.ntuple.root\"")

# ROOT-6286
ROOTTEST_ADD_TEST(cleanupChain
                  MACRO runcleanupChain.C)

ROOTTEST_ADD_TEST(hsimple-create
                  MACRO ${CMAKE_SOURCE_DIR}/tutorials/hsimple.C
                  PASSRC 255
                  FIXTURES_SETUP root-tree-reader-hsimple-fixture)

ROOTTEST_ADD_TEST(ReaderIterator
                  MACRO assertReaderIterator.C
                  FIXTURES_REQUIRED root-tree-reader-hsimple-fixture)

ROOTTEST_ADD_TEST(Structlong64
                  MACRO assertStructlong64.C)

ROOTTEST_ADD_TEST(CheckLateProxy
                  MACRO execCheckLateProxy.C
                  OUTREF execCheckLateProxy.ref
                  FIXTURES_REQUIRED root-tree-reader-hsimple-fixture)

ROOTTEST_ADD_TEST(Enum
                  MACRO execEnum.cxx+
                  OUTREF execEnum.ref)

ROOTTEST_ADD_TEST(VectorBranches
                  MACRO execVectorBranches.C
                  OUTREF execVectorBranches.ref)

ROOTTEST_COMPILE_MACRO(A.h
                       FIXTURES_SETUP root-tree-reader-A-fixture)

ROOTTEST_COMPILE_MACRO(hardTreeReaderTest.cpp
                       BUILDLIB A_h
                       FIXTURES_REQUIRED root-tree-reader-A-fixture
                       FIXTURES_SETUP root-tree-reader-hardTreeReaderTest-fixture)

ROOTTEST_ADD_TEST(complexTree
                  MACRO runcomplexTree.C
                  OUTREF complexTree.ref
                  FIXTURES_REQUIRED root-tree-reader-A-fixture root-tree-reader-hardTreeReaderTest-fixture)

# tests reading files from EOS and requires xrootd enabled,
if(xrootd)
  ROOTTEST_ADD_TEST(assertIntroTut
                    MACRO assertIntroTut.C
                    ERRREF assertIntroTut.ref)

  ROOTTEST_ADD_TEST(IntroTut
                    MACRO execIntroTut.C
                    OUTREF execIntroTut.ref)
endif()
