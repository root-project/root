if(${compression_default} STREQUAL "lz4")
      ROOTTEST_ADD_TEST(multiTree
                  MACRO runmultiTree.C
                  OUTREF multiTreeLZ4.ref)

      ROOTTEST_ADD_TEST(autocache
                  COPY_TO_BUILDDIR AliESDs-0.root AliESDs-1.root
                  MACRO runautocache.C
                  OUTREF autocacheLZ4.ref
                  ERRREF autocache.eref)
else(${compression_default} STREQUAL "zlib")
      ROOTTEST_ADD_TEST(multiTree
                  MACRO runmultiTree.C
                  OUTREF multiTreeZLIB.ref)
      if(ZLIB_CF)
         ROOTTEST_ADD_TEST(autocache
                  COPY_TO_BUILDDIR AliESDs-0.root AliESDs-1.root
                  MACRO runautocache.C
                  OUTREF autocacheZLIB_builtinzlib.ref
                  ERRREF autocache.eref)
      else()
         ROOTTEST_ADD_TEST(autocache
                  COPY_TO_BUILDDIR AliESDs-0.root AliESDs-1.root
                  MACRO runautocache.C
                  OUTREF autocacheZLIB.ref
                  ERRREF autocache.eref)
     endif()
endif()

ROOTTEST_ADD_TEST(cachedChain
                  MACRO runcachedchain.C
                  OUTCNVCMD grep -v -e "no dictionary for class" -e "trying to read an emulated class (AliESDEvent)"
                  OUTREF cachedchain.ref
                  DEPENDS autocache)

ROOTTEST_ADD_TEST(TestCache
                  MACRO execTestCache.cxx+
                  OUTREF execTestCache.ref)

ROOTTEST_ADD_TEST(TestMissCache
                  MACRO execTestMissCache.cxx+
                  OUTREF execTestMissCache.ref)

ROOTTEST_ADD_TEST(TooSmall
                  MACRO assertTooSmall.C
                  ERRREF assertTooSmall.eref)

ROOTTEST_ADD_TEST(CacheRange
                  MACRO execCacheRange.C
                  OUTREF execCacheRange.ref)

ROOTTEST_ADD_TEST(LastCluster
                  COPY_TO_BUILDDIR lastcluster.root
                  MACRO execLastCluster.C
                  OUTREF execLastCluster.ref)

ROOT_GENERATE_DICTIONARY(G__TheEvent ${CMAKE_CURRENT_SOURCE_DIR}/Event.h LINKDEF EventLinkDef.h)
ROOTTEST_LINKER_LIBRARY(TheEvent TEST Event.cxx G__TheEvent.cxx LIBRARIES ${ROOT_LIBRARIES})
ROOTTEST_ADD_TEST(perfstattest-libevent-build 
                  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --config $<CONFIG> --target G__TheEvent${fast} TheEvent${fast} -- ${always-make})

set(RootExeOptions -e "gSystem->Load(\"libTheEvent\")")
if(compression_default STREQUAL "lz4")
      ROOTTEST_ADD_TEST(perfstattest
                  MACRO execperfstattest.C
                  OUTREF execperfstattestLZ4.oref
                  ERRREF execperfstattest.eref
                  DEPENDS perfstattest-libevent-build)
else()
      ROOTTEST_ADD_TEST(perfstattest
                  MACRO execperfstattest.C
                  OUTREF execperfstattestZLIB.oref
                  ERRREF execperfstattest.eref
                  DEPENDS perfstattest-libevent-build)
endif()
