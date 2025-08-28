# ROOT-5983
ROOTTEST_ADD_TEST(initOffset
                  MACRO  initOffset.C+)

ROOTTEST_ADD_TEST(abc_rflx_gen_dict
                  COMMAND ${ROOT_genreflex_CMD} ${CMAKE_CURRENT_SOURCE_DIR}/abc.h -o standalone_abc_rflx.cpp)

ROOTTEST_COMPILE_MACRO(abc.h
                       FIXTURES_SETUP root-tree-branches-abc-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(abc_rflx abc.h
                                    SELECTION selection.xml
                                    LIBNAME libabc_rflx
                                    NO_ROOTMAP
                                    FIXTURES_SETUP root-tree-branches-libabc_rflx-fixture)

set(_opts_cint -e "(void)gROOT->ProcessLine(\".L abc.h+\")")
set(_opts_rflx -e "(void)gSystem->Load(\"libabc_rflx\")")

ROOTTEST_ADD_TEST(abc-cint-write
                  MACRO abcwrite.C
                  MACROARG "\"cint\""
                  ROOTEXE_OPTS ${_opts_cint}
                  FIXTURES_REQUIRED root-tree-branches-abc-fixture
                  FIXTURES_SETUP root-tree-branches-abc-cint-write-fixture)

ROOTTEST_ADD_TEST(abc-reflex-write
                  MACRO abcwrite.C
                  MACROARG "\"reflex\""
                  ROOTEXE_OPTS ${_opts_rflx}
                  FIXTURES_REQUIRED root-tree-branches-libabc_rflx-fixture
                  FIXTURES_SETUP root-tree-branches-abc-reflex-write-fixture)

ROOTTEST_ADD_TEST(abc-cint-cint-read
                  MACRO abcread.C
                  MACROARG "\"cint\",\"cint\""
                  ROOTEXE_OPTS ${_opts_cint}
                  OUTREF abc.ref
                  FIXTURES_REQUIRED root-tree-branches-abc-cint-write-fixture
                                    root-tree-branches-abc-fixture)

ROOTTEST_ADD_TEST(abc-cint-reflex-read
                  MACRO abcread.C
                  MACROARG "\"cint\",\"reflex\""
                  ROOTEXE_OPTS ${_opts_rflx}
                  OUTREF abc.ref
                  FIXTURES_REQUIRED root-tree-branches-abc-cint-write-fixture
                                    root-tree-branches-libabc_rflx-fixture)

ROOTTEST_ADD_TEST(abc-reflex-cint-read
                  MACRO abcread.C
                  MACROARG "\"reflex\",\"cint\""
                  ROOTEXE_OPTS ${_opts_cint}
                  OUTREF abc.ref
                  FIXTURES_REQUIRED root-tree-branches-abc-reflex-write-fixture
                                    root-tree-branches-abc-fixture)

ROOTTEST_ADD_TEST(abc-reflex-reflex-read
                  MACRO abcread.C
                  MACROARG "\"reflex\",\"reflex\""
                  ROOTEXE_OPTS ${_opts_rflx}
                  OUTREF abc.ref
                  FIXTURES_REQUIRED root-tree-branches-abc-reflex-write-fixture
                                    root-tree-branches-libabc_rflx-fixture)

ROOTTEST_ADD_TEST(MakeClass
                  MACRO  assertMakeClass.C
                  ERRREF assertMakeClass.ref)

ROOTTEST_ADD_TEST(bref
                  MACRO  execbref.C
                  OUTREF execbref.ref)

ROOTTEST_ADD_TEST(DupNames
                  MACRO  execDupNames.cxx+
                  OUTREF execDupNames.ref)

ROOTTEST_ADD_TEST(EmptyBase
                  MACRO  execEmptyBase.cxx+
                  OUTREF execEmptyBase.ref)

ROOTTEST_ADD_TEST(GetBranch
                  MACRO  execGetBranch.cxx+
                  OUTREF execGetBranch.ref)

ROOTTEST_ADD_TEST(LeafFullName
                  MACRO  execLeafFullName.cxx+
                  OUTREF execLeafFullName.ref)

ROOTTEST_ADD_TEST(WriteRead
                  MACRO  execWriteRead.C
                  OUTREF execWriteRead.ref)

ROOTTEST_ADD_TEST(simple
                  MACRO  runsimple.C+
                  OUTREF simple.ref)

ROOTTEST_ADD_TEST(i_simple
                  MACRO  runsimple.C
                  OUTREF i_simple.ref)

ROOTTEST_ADD_TEST(noname
                  MACRO  runnoname.C
                  OUTREF noname.ref)

# on windows test fails while after reading of treeobj.root file branch.root left open
if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(separate
                    MACRO  runseparate.C
                    OUTREF separate.ref)
endif()

ROOTTEST_ADD_TEST(updates
                  MACRO  runupdates.C
                  OUTREF updates.ref)
