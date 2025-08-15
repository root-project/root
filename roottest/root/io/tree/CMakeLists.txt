ROOTTEST_COMPILE_MACRO(classes.C
                       FIXTURES_SETUP root-io-tree-classes-fixture)

ROOTTEST_ADD_TEST(write
                  MACRO write.C
                  ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L classes.C+\")"
                  FIXTURES_REQUIRED root-io-tree-classes-fixture
                  FIXTURES_SETUP root-io-tree-write-fixture)

ROOTTEST_ADD_TEST(copy
                  MACRO copy.C
                  ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L classes.C+\")"
                  FIXTURES_REQUIRED root-io-tree-classes-fixture
                                    root-io-tree-write-fixture
                  FIXTURES_SETUP root-io-tree-copy-fixture)

ROOTTEST_ADD_TEST(read
                  MACRO read.C
                  OUTREF missingClass.ref
                  FIXTURES_REQUIRED root-io-tree-copy-fixture)

ROOTTEST_ADD_TEST(tref-write
                  MACRO tref_test.C
                  FIXTURES_SETUP root-io-tree-tref-write-fixture)

ROOTTEST_ADD_TEST(tref-read
                  MACRO tref_test_rd.C
                  OUTREF reftest.ref
                  FIXTURES_REQUIRED root-io-tree-tref-write-fixture)

ROOTTEST_COMPILE_MACRO(tref_test_pid.C
                       FIXTURES_SETUP root-io-tree-macro-fixture)

function(ROOTTEST_IO_TREE_TEST name require outref)
  if(require)
    set(_fixture root-io-tree-${require}-fixture)
  endif()

  ROOTTEST_ADD_TEST(${name}
                    ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L tref_test_pid.C+\")"
                               -e "#define STAGE \"${name}\""
                    MACRO run.C
                    OUTREF ${outref}
                    FIXTURES_REQUIRED root-io-tree-macro-fixture ${_fixture}
                    FIXTURES_SETUP root-io-tree-${name}-fixture)
endfunction(ROOTTEST_IO_TREE_TEST)


ROOTTEST_IO_TREE_TEST(first "" "")
ROOTTEST_IO_TREE_TEST(create-r first "")
ROOTTEST_IO_TREE_TEST(test-r create-r test-r.ref)
ROOTTEST_IO_TREE_TEST(create-rc create-r "")
ROOTTEST_IO_TREE_TEST(test-rc create-rc test-rc.ref)
ROOTTEST_IO_TREE_TEST(create-rcr create-rc "")
ROOTTEST_IO_TREE_TEST(test-rcr create-rcr test-rcr.ref)
ROOTTEST_IO_TREE_TEST(test-rcr2 create-rcr test-rcr2.ref)
ROOTTEST_IO_TREE_TEST(create-rcrr create-rcr "")
ROOTTEST_IO_TREE_TEST(test-rcrr create-rcrr test-rcrr.ref)
ROOTTEST_IO_TREE_TEST(test-rcrr2 create-rcrr test-rcrr2.ref)
ROOTTEST_IO_TREE_TEST(test-rcrr3 create-rcrr test-rcrr3.ref)
ROOTTEST_IO_TREE_TEST(test-rcrr123 create-rcrr test-rcrr123.ref)
