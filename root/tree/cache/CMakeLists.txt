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
