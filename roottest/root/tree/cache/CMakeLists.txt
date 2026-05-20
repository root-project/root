ROOTTEST_ADD_TEST(multiTree
          MACRO runmultiTree.C
          OUTREF multiTree.ref)

if(ZLIB_NG)
  ROOTTEST_ADD_TEST(autocache
            COPY_TO_BUILDDIR AliESDs-0.root AliESDs-1.root
            MACRO runautocache.C
            OUTREF autocacheZLIB-NG.ref
            ERRREF autocache.eref)
else()
  ROOTTEST_ADD_TEST(autocache
            COPY_TO_BUILDDIR AliESDs-0.root AliESDs-1.root
            MACRO runautocache.C
            OUTREF autocacheZLIB.ref
            ERRREF autocache.eref)
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

ROOTTEST_ADD_TEST(perfstattest
                  MACRO execperfstattest.C
                  ROOTEXE_OPTS -e "gInterpreter->AddIncludePath(\"${CMAKE_SOURCE_DIR}/roottest/root/io/event/\")" -e "gSystem->AddDynamicPath(\"${CMAKE_BINARY_DIR}/roottest/root/io/event/\")" -e "(void)gSystem->Load(\"libTestIoEvent\")"
                  OUTREF execperfstattest.oref
                  ERRREF execperfstattest.eref
                  FIXTURES_REQUIRED root-io-event-libTestIoEvent-fixture)
