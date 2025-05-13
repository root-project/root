
ROOTTEST_GENERATE_DICTIONARY(libTreeProblemFoo
                  Foo.h Foo.cxx
                  LINKDEF LinkDef.h
                  NO_ROOTMAP NO_CXXMODULE
                  FIXTURES_SETUP TreeProblemFoo_fixture)

ROOTTEST_ADD_TEST(writer
                  MACRO writer.C
                  OUTREF writer.ref
                  FIXTURES_REQUIRED TreeProblemFoo_fixture
                  FIXTURES_SETUP treeproblem_writer)

ROOTTEST_ADD_TEST(reader
                  MACRO reader.C
                  OUTREF reader.ref
                  COPY_TO_BUILDDIR tree.h tree.C
                  FIXTURES_REQUIRED treeproblem_writer)
