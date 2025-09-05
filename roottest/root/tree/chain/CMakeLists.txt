ROOTTEST_ADD_TEST(ChainElementStatus
                  MACRO execChainElementStatus.C
                  OUTREF execChainElementStatus.ref
                  OUTCNVCMD sed -e "s?: file [A-Za-z/\].*[/\]?: file ?"
                  COPY_TO_BUILDDIR st4.root st-empty.root st2.root st-notree.root st8.root)

ROOTTEST_ADD_TEST(Cleanup
                  MACRO execCleanup.C
                  OUTREF execCleanup.ref)

ROOTTEST_ADD_TEST(Empty
                  MACRO execEmpty.C
                  OUTREF execEmpty.ref
                  COPY_TO_BUILDDIR n_empty.root n_full.root)

ROOTTEST_ADD_TEST(filenameformats
                  MACRO execfilenameformats.C
                  OUTREF execfilenameformats.ref)

ROOTTEST_ADD_TEST(missingIndex
                  MACRO execmissingIndex.C
                  OUTREF execmissingIndex.ref)

ROOTTEST_ADD_TEST(Notify
                  MACRO execNotify.C
                  OUTREF execNotify.ref)

ROOTTEST_ADD_TEST(resetbranch
                  MACRO execresetbranch.C
                  OUTREF execresetbranch.ref)

ROOTTEST_ADD_TEST(TChainCleanup
                  MACRO execTChainCleanup.cxx+
                  OUTREF execTChainCleanup.ref)

# to avoid copying of ROOT files - just run from source directory
ROOTTEST_ADD_TEST(subdir
                  MACRO runsubdir.C
                  WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTREF subdir.ref)
