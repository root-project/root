ROOTTEST_COMPILE_MACRO(stringarray.h
                       FIXTURES_SETUP root-io-evolution-issue-8083-stringarray-fixture)

ROOTTEST_COMPILE_MACRO(writefile.cxx
                       FIXTURES_SETUP root-io-evolution-issue-8083-writefile-fixture)

ROOTTEST_ADD_TEST(WriteAfterOld
                  MACRO execWriteAfterOld.C+
                  COPY_TO_BUILDDIR stringarray.old.root
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-issue-8083-stringarray-fixture root-io-evolution-issue-8083-writefile-fixture
                  FIXTURES_SETUP root-io-evolution-issue-8083-WriteAfterOld-fixture)

ROOTTEST_ADD_TEST(readfile
                  MACRO execreadfile.cxx+
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-issue-8083-WriteAfterOld-fixture)
