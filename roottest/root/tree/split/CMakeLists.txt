ROOTTEST_ADD_TEST(abstract-copy
                  COMMAND ${CMAKE_COMMAND} -E echo "copy abstract.cxx"
                  COPY_TO_BUILDDIR abstract.cxx
                  FIXTURES_SETUP root-tree-split-abstract-copy-fixture)

ROOTTEST_COMPILE_MACRO(abstract.cxx
                       FIXTURES_REQUIRED root-tree-split-abstract-copy-fixture
                       FIXTURES_SETUP root-tree-split-abstract-fixture)

ROOTTEST_ADD_TEST(abstract-write
                  MACRO runabstract.cxx
                  MACROARG 0
                  FIXTURES_REQUIRED root-tree-split-abstract-fixture
                  FIXTURES_SETUP root-tree-split-abstract-write-fixture)

ROOTTEST_ADD_TEST(abstract-read
                  MACRO runabstract.cxx
                  MACROARG 1
                  OUTREF abstract.ref
                  FIXTURES_REQUIRED root-tree-split-abstract-fixture
                                    root-tree-split-abstract-write-fixture)

ROOTTEST_COMPILE_MACRO(tobj.cxx
                       FIXTURES_SETUP root-tree-split-tobj-fixture)

ROOTTEST_ADD_TEST(CollectionSplit
                  MACRO tobj.cxx+
                  OUTREF CollectionSplit.ref
                  OUTCNVCMD grep Br
                  FIXTURES_REQUIRED root-tree-split-tobj-fixture)

ROOTTEST_ADD_TEST(WriteCustomStream
                  MACRO execWriteCustomStream.cxx+
                  OUTREF execWriteCustomStream.ref
                  FIXTURES_SETUP root-tree-split-WriteCustomStream-fixture)

ROOTTEST_ADD_TEST(ReadCustomStream
                  MACRO execReadCustomStream.cxx+
                  OUTREF execReadCustomStream.ref
                  FIXTURES_REQUIRED root-tree-split-WriteCustomStream-fixture)

ROOTTEST_ADD_TEST(UnrollWithConst
                  MACRO execUnrollWithConst.cxx+
                  OUTREF execUnrollWithConst.ref)
