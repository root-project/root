ROOTTEST_GENERATE_DICTIONARY(libFoo
                             Foo.h
                             SOURCES Foo.cxx
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-tree-treeproblem-Foo)

ROOTTEST_ADD_TEST(writer
                  MACRO writer.C
                  FIXTURES_REQUIRED root-tree-treeproblem-Foo
                  FIXTURES_SETUP root-tree-treeproblem-writer)

ROOTTEST_ADD_TEST(reader
                  MACRO reader.C
                  OUTREF reader.ref
                  FIXTURES_REQUIRED root-tree-treeproblem-writer)

ROOTTEST_ADD_TEST(ucharshow
                  MACRO runucharshow.C
                  OUTREF ucharshow.ref)

ROOTTEST_ADD_TEST(Memory
                  MACRO runMemory.C
                  OUTREF Memory.ref)

ROOTTEST_ADD_TEST(Scan
                  MACRO execScan.C
                  OUTREF execScan.ref)