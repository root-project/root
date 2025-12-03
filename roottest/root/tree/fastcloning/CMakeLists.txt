if(${compression_default} STREQUAL "zlib")
   if(CMAKE_SIZEOF_VOID_P EQUAL 4)
      # 32 bits
      ROOTTEST_ADD_TEST(execCheckClusterRange
                  MACRO execCheckClusterRange.C
                  OUTREF references/execCheckClusterRange.ref32)
   else()
      # 64 bits
      if(ZLIB_CF)
         ROOTTEST_ADD_TEST(execCheckClusterRange
                  MACRO execCheckClusterRange.C
                  OUTREF references/execCheckClusterRange_builtinzlib.ref)
      elseif(MSVC)
         ROOTTEST_ADD_TEST(execCheckClusterRange
                  MACRO execCheckClusterRange.C
                  OUTREF references/execCheckClusterRange_zlib_win64.ref)
      else()
         ROOTTEST_ADD_TEST(execCheckClusterRange
                  MACRO execCheckClusterRange.C
                  OUTREF references/execCheckClusterRange_zlib.ref)
      endif()
   endif()
endif()

ROOTTEST_ADD_TEST(execmisstop
                  MACRO execmisstop.C
                  OUTREF references/execmisstop.ref)

ROOTTEST_COMPILE_MACRO(abstract.C
                       FIXTURES_SETUP root-tree-fastcloning-abstract-fixture)

ROOTTEST_ADD_TEST(runabstract-datageneration0
                  MACRO abstract.C+
                  MACROARG 0
                  FIXTURES_REQUIRED root-tree-fastcloning-abstract-fixture
                  FIXTURES_SETUP root-tree-fastcloning-runabstract-datageneration0-fixture)

ROOTTEST_ADD_TEST(runabstract-datageneration1
                  MACRO abstract.C+
                  MACROARG 1
                  FIXTURES_REQUIRED root-tree-fastcloning-abstract-fixture
                                    root-tree-fastcloning-runabstract-datageneration0-fixture
                  FIXTURES_SETUP root-tree-fastcloning-runabstract-datageneration1-fixture)

ROOTTEST_ADD_TEST(runabstract-copy
                  MACRO runabstract.C
                  OUTREF references/abstract.ref
                  FIXTURES_REQUIRED root-tree-fastcloning-runabstract-datageneration1-fixture)

ROOTTEST_ADD_TEST(runfilemergererror
                  MACRO runfilemergererror.C
                  OUTREF references/runfilemergererror.ref)

ROOTTEST_ADD_TEST(runoutoforder-datageneration
                  COPY_TO_BUILDDIR files/Tuple_1.root files/Tuple_2.root
                  COMMAND ${ROOT_hadd_CMD} -f Tuple_merge.root Tuple_1.root Tuple_2.root
                  FIXTURES_SETUP root-tree-fastcloning-runoutoforder-datageneration-fixture)

ROOTTEST_ADD_TEST(runoutoforder
                  MACRO runoutoforder.C
                  OUTREF references/runoutoforder.ref
                  FIXTURES_REQUIRED root-tree-fastcloning-runoutoforder-datageneration-fixture)

ROOTTEST_ADD_TEST(runPlot
                  MACRO runPlot.C
                  COPY_TO_BUILDDIR files/recon-v1r030603p6_700000811_00000-00984_calTuple.root
                                   files/recon-v1r030603p6_700000811_02955-03939_calTuple.root
                  OUTREF references/runPlot.ref)

ROOTTEST_ADD_TEST(runbadmix
                  MACRO runbadmix.C
                  OUTREF references/runbadmix.ref)

ROOTTEST_ADD_TEST(runSplitMismatch
                  MACRO runSplitMismatch.C+
                  OUTREF references/runSplitMismatch.ref)

ROOTTEST_ADD_TEST(make_CloneTree
                  MACRO make_CloneTree.C
                  COPY_TO_BUILDDIR files/ver_40200.root files/ver_40200_copy.root
                  FIXTURES_SETUP root-tree-fastcloning-makeCloneTree-fixture)

ROOTTEST_ADD_TEST(runcloneChain
                  MACRO runcloneChain.C
                  OUTREF references/runcloneChain.ref
                  FIXTURES_REQUIRED root-tree-fastcloning-makeCloneTree-fixture)
