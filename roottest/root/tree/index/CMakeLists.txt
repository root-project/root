ROOTTEST_ADD_TEST(chain
                  MACRO runchain.C
                  OUTREF chain.ref)

ROOTTEST_ADD_TEST(ChainIndex
                  MACRO execChainIndex.cxx+
                  COPY_TO_BUILDDIR abarton.1.root abarton.2.root abarton.3.root
                  OUTREF execChainIndex.ref)

ROOTTEST_ADD_TEST(Dupe
                  MACRO runDupe.C
                  OUTREF Dupe.ref)

ROOTTEST_ADD_TEST(index64
                  MACRO runindex64.C
                  OUTREF index64.ref)

