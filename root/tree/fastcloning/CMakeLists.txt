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

ROOTTEST_ADD_TEST(runabstract-datageneration
                  COPY_TO_BUILDDIR abstract.C
                  COMMAND ${ROOT_root_CMD} -b -q -l "${CMAKE_CURRENT_BINARY_DIR}/abstract.C+(0)")

ROOTTEST_ADD_TEST(runabstract-copy
                  MACRO runabstract.C
                  PRECMD ${ROOT_root_CMD} -b -q -l "${CMAKE_CURRENT_BINARY_DIR}/abstract.C+(1)"
                  OUTREF references/abstract.ref
                  DEPENDS runabstract-datageneration)

ROOTTEST_ADD_TEST(runfilemergererror
                  MACRO runfilemergererror.C
                  OUTREF references/runfilemergererror.ref)

ROOTTEST_ADD_TEST(runoutoforder-datageneration
                  COPY_TO_BUILDDIR files/Tuple_1.root
                  COPY_TO_BUILDDIR files/Tuple_2.root
                  COMMAND ${ROOT_hadd_CMD} -f Tuple_merge.root Tuple_1.root Tuple_2.root)

ROOTTEST_ADD_TEST(runoutoforder
                  MACRO runoutoforder.C
                  OUTREF references/runoutoforder.ref
                  DEPENDS runoutoforder-datageneration)

ROOTTEST_ADD_TEST(runPlot
                  MACRO runPlot.C
                  COPY_TO_BUILDDIR files/recon-v1r030603p6_700000811_00000-00984_calTuple.root
                  COPY_TO_BUILDDIR files/recon-v1r030603p6_700000811_02955-03939_calTuple.root
                  OUTREF references/runPlot.ref)

ROOTTEST_ADD_TEST(runbadmix
                  MACRO runbadmix.C
                  PRECMD ${ROOT_root_CMD} -b -q -l ${CMAKE_CURRENT_SOURCE_DIR}/runbadmix.C
                  OUTREF references/runbadmix.ref)

ROOTTEST_ADD_TEST(runSplitMismatch
                  COPY_TO_BUILDDIR runSplitMismatch.C
                  MACRO ${CMAKE_CURRENT_BINARY_DIR}/runSplitMismatch.C+
                  OUTREF references/runSplitMismatch.ref)

ROOTTEST_ADD_TEST(make_CloneTree
                  MACRO make_CloneTree.C
                  COPY_TO_BUILDDIR files/ver_40200.root
                  COPY_TO_BUILDDIR files/ver_40200_copy.root)

ROOTTEST_ADD_TEST(runcloneChain
                  MACRO runcloneChain.C
                  OUTREF references/runcloneChain.ref
                  DEPENDS roottest-root-tree-fastcloning-make_CloneTree)
