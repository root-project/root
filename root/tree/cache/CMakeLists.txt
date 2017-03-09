ROOTTEST_ADD_TEST(multiTree
                  MACRO runmultiTree.C
                  OUTREF multiTree.ref)

ROOTTEST_ADD_TEST(autocache
                  COPY_TO_BUILDDIR AliESDs-0.root AliESDs-1.root
                  MACRO runautocache.C
                  OUTREF autocache.ref
                  ERRREF autocache.eref)

ROOTTEST_ADD_TEST(cachedChain
                  MACRO runcachedchain.C
                  OUTCNVCMD grep -v -e "no dictionary for class" -e "trying to read an emulated class (AliESDEvent)"
                  OUTREF cachedchain.ref
                  DEPENDS autocache)

ROOTTEST_ADD_TEST(TestCache
                  MACRO execTestCache.cxx+
                  OUTREF execTestCache.ref)
                  
ROOT_GENERATE_DICTIONARY(TheEventDict Event.h LINKDEF EventLinkDef.h)
ROOT_LINKER_LIBRARY(TheEvent TEST Event.cxx TheEventDict.cxx LIBRARIES ${ROOT_LIBRARIES})
ROOTTEST_ADD_TEST(perfstattest-libevent-build 
                  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target TheEvent${fast})

set(RootExeOptions -e "gSystem->Load(\"libTheEvent\")")
ROOTTEST_ADD_TEST(perfstattest
                  MACRO execperfstattest.C
                  OUTREF execperfstattest.oref
                  ERRREF execperfstattest.eref
                  DEPENDS perfstattest-libevent-build)

