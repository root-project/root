if(xrootd)
  ROOTTEST_GENERATE_DICTIONARY(libRoottestIoPrefetching
                               atlasFlushed/atlasFlushedProjectHeaders.h
                               LINKDEF atlasFlushed/atlasFlushedLinkDef.h
                               SOURCES atlasFlushed/atlasFlushedProjectSource.cxx
                               NO_CXXMODULE NO_ROOTMAP
                               FIXTURES_SETUP root-io-prefetching-dict-fixture)
  ROOTTEST_ADD_TEST(run
                    MACRO runPrefetchReading.C+
                    OUTREF PrefetchReading.ref
                    LABELS longtest
                    FIXTURES_REQUIRED root-io-prefetching-dict-fixture)
endif()