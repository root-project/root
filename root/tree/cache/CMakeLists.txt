ROOTTEST_ADD_TEST(roottest-root-tree-cache-multiTree
                  MACRO runmultiTree.C
                  OUTREF multiTree.ref)

ROOTTEST_ADD_TEST(roottest-root-tree-cache-autocache
                  COPY_TO_BUILDDIR AliESDs-0.root AliESDs-1.root
                  MACRO runautocache.C
                  OUTCNVCMD grep -v -e "no dictionary for class" -e "trying to read an emulated class (AliESDEvent)"
                  OUTREF autocache.ref)

ROOTTEST_ADD_TEST(roottest-root-tree-cache-cachedChain
                  MACRO runcachedchain.C
                  OUTCNVCMD grep -v -e "no dictionary for class" -e "trying to read an emulated class (AliESDEvent)"
                  OUTREF cachedchain.ref
                  DEPENDS roottest-root-tree-cache-autocache)

ROOTTEST_ADD_TEST(roottest-root-tree-cache-TestCache
                  MACRO execTestCache.cxx+
                  OUTREF execTestCache.ref)
