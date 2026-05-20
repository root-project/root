ROOTTEST_GENERATE_DICTIONARY(ExtHit
                             ExtHit/ExtHitProjectHeaders.h
                             LINKDEF ExtHit/ExtHitLinkDef.h
                             SOURCES ExtHit/ExtHitProjectSource.cxx
                             NO_ROOTMAP NO_CXXMODULE
                             FIXTURES_SETUP root-tree-addresses-convertIssues-ExtHit-fixture)

target_link_libraries(ExtHitlibgen Physics)

ROOTTEST_ADD_TEST(ROOT8794
                  MACRO execROOT8794.C
                  OUTREF execROOT8794.ref
                  COPY_TO_BUILDDIR belle2-ROOT-8794.root
                  FIXTURES_REQUIRED root-tree-addresses-convertIssues-ExtHit-fixture)

ROOTTEST_COMPILE_MACRO(execWriter.cxx
                       FIXTURES_SETUP root-tree-addresses-convertIssues-execWriterCompile-fixture)

ROOTTEST_ADD_TEST(Writer
                  MACRO execWriter.cxx+
                  OUTREF execWriter.ref
                  FIXTURES_REQUIRED root-tree-addresses-convertIssues-execWriterCompile-fixture
                  FIXTURES_SETUP root-tree-addresses-convertIssues-Writer-fixture)

ROOTTEST_ADD_TEST(Reader
                  MACRO execReader.cxx+
                  OUTREF execReader.ref
                  FIXTURES_REQUIRED root-tree-addresses-convertIssues-Writer-fixture)
