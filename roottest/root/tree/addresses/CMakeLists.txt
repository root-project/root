ROOTTEST_ADD_TESTDIRS()

ROOTTEST_COMPILE_MACRO(inherit.C
                       FIXTURES_SETUP root-tree-addresses-inherit-fixture)

if(ClingWorkAroundMissingDynamicScope)
  SET(_inherit_opts -e "(void)gROOT->ProcessLine(\".L inherit.C+g\")")
endif()

ROOTTEST_ADD_TEST(inherit
                  MACRO runInherit.C
                  ROOTEXE_OPTS ${_inherit_opts}
                  OUTREF inheritTest.ref
                  FIXTURES_REQUIRED root-tree-addresses-inherit-fixture)

ROOTTEST_COMPILE_MACRO(header.h
                       FIXTURES_SETUP root-tree-addresses-header-fixture)

if(ClingWorkAroundMissingDynamicScope)
  SET(_header_opts -e "(void)gROOT->ProcessLine(\".L header.h+g\")")
endif()

ROOTTEST_ADD_TEST(header
                  MACRO runHeader.C
                  ROOTEXE_OPTS ${_header_opts}
                  OUTREF headerTest.ref
                  FIXTURES_REQUIRED root-tree-addresses-header-fixture)

ROOTTEST_COMPILE_MACRO(clonesA_Event.cxx
                       FIXTURES_SETUP root-tree-addresses-clonesA_Event-fixture)

if(ClingWorkAroundMissingDynamicScope)
  SET(_clonesA_opts -e "(void)gROOT->ProcessLine(\".L clonesA_Event.cxx+\")")
endif()

ROOTTEST_ADD_TEST(clonesA
                  MACRO clonesA_rw.C
                  ROOTEXE_OPTS ${_clonesA_opts}
                  FIXTURES_REQUIRED root-tree-addresses-clonesA_Event-fixture)

ROOTTEST_COMPILE_MACRO(Embedded_load.C
                       FIXTURES_SETUP root-tree-addresses-Embedded_load-fixture)

if(ClingWorkAroundMissingDynamicScope)
  SET(_embedLoad_opts -e "(void)gROOT->ProcessLine(\".L Embedded_load.C+\")")
endif()

ROOTTEST_ADD_TEST(embwrite
                  MACRO embwrite.C
                  ROOTEXE_OPTS ${_embedLoad_opts}
                  FIXTURES_REQUIRED root-tree-addresses-Embedded_load-fixture
                  FIXTURES_SETUP root-tree-addresses-embwrite-fixture)

ROOTTEST_ADD_TEST(embread
                  MACRO embread.C
                  ROOTEXE_OPTS ${_embedLoad_opts}
                  OUTREF EmbeddedTest.ref
                  FIXTURES_REQUIRED root-tree-addresses-Embedded_load-fixture
                                    root-tree-addresses-embwrite-fixture)

# disable on Windows because relations_write.C compilation problem
# in any case test cannot be run on Windows because of sed usage
if(NOT MSVC OR win_broken_tests)
  ROOTTEST_COMPILE_MACRO(relations_load.C
                         FIXTURES_SETUP root-tree-addresses-relations_load-fixture)

  ROOTTEST_COMPILE_MACRO(relations_read.C
                         FIXTURES_SETUP root-tree-addresses-relations_read-fixture)

  ROOTTEST_COMPILE_MACRO(relations_write.C
                         FIXTURES_SETUP root-tree-addresses-relations_write-fixture)

  SET(_relations_opts -e "(void)gROOT->ProcessLine(\".L relations_load.C+\")"
                      -e "(void)gROOT->ProcessLine(\".L relations_write.C+\")"
                      -e "(void)gROOT->ProcessLine(\".L relations_read.C+\")")

  ROOTTEST_ADD_TEST(relations
                    MACRO relationsRun.C
                    ROOTEXE_OPTS ${_relations_opts}
                    OUTREF relationsTest.ref
                    OUTCNVCMD sed -e "s|File  Size =.*\\*||g"
                    FIXTURES_REQUIRED root-tree-addresses-relations_load-fixture
                                      root-tree-addresses-relations_read-fixture
                                      root-tree-addresses-relations_write-fixture)
endif()

# disable on Windows because of problem with running same macro twice
if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(merging
                    MACRO runmerging.C
                    OUTREF merging.ref)
endif()

ROOTTEST_COMPILE_MACRO(iobug.C
                       FIXTURES_SETUP root-tree-addresses-iobug-fixture)

ROOTTEST_ADD_TEST(iobug
                  MACRO runiobug.C
                  ROOTEXE_OPTS -e "(void)gSystem->Load(\"../../../utils_cc\")"
                  OUTREF iobug.ref
                  FIXTURES_REQUIRED root-tree-addresses-iobug-fixture roottest-scripts-utils-fixture)

ROOTTEST_COMPILE_MACRO(userClass.C
                       FIXTURES_SETUP root-tree-addresses-userClass-fixture)

ROOTTEST_COMPILE_MACRO(treeBranch.C
                       FIXTURES_SETUP root-tree-addresses-treeBranch-fixture)

if(ClingWorkAroundMissingUnloading)
  set(_treeBranch_ref treeBranch.ref-missingUnloading)
else()
  set(_treeBranch_ref treeBranch.ref)
endif()

ROOTTEST_ADD_TEST(treeBranch
                  MACRO runtreeBranch.C
                  OUTREF ${_treeBranch_ref}
                  FIXTURES_REQUIRED root-tree-addresses-userClass-fixture
                                    root-tree-addresses-treeBranch-fixture)

ROOTTEST_COMPILE_MACRO(test_Persistency0.C
                       FIXTURES_SETUP root-tree-addresses-test_Persistency0-fixture)

ROOTTEST_COMPILE_MACRO(test_Persistency1.C
                       FIXTURES_SETUP root-tree-addresses-test_Persistency1-fixture)

ROOTTEST_COMPILE_MACRO(test_Persistency3.C
                       FIXTURES_SETUP root-tree-addresses-test_Persistency3-fixture)

ROOTTEST_ADD_TEST(inheritAndOwn
                  MACRO runinheritAndOwn.C
                  OUTREF inheritAndOwn.ref
                  FIXTURES_REQUIRED root-tree-addresses-test_Persistency0-fixture
                                    root-tree-addresses-test_Persistency3-fixture)

ROOTTEST_ADD_TEST(BadBaseAddresses
                  MACRO runBadBaseAddresses.C
                  OUTREF BadBaseAddresses.ref
                  FIXTURES_REQUIRED root-tree-addresses-test_Persistency1-fixture)

ROOTTEST_ADD_TEST(ursula
                  MACRO runursula.C
                  COPY_TO_BUILDDIR cmsursula.root
                  OUTREF ursula.ref)

ROOTTEST_COMPILE_MACRO(sueloader.C
                       FIXTURES_SETUP root-tree-addresses-sueloader-fixture)

ROOTTEST_COMPILE_MACRO(ConfigRecord.cxx
                       BUILDLIB sueloader_C
                       FIXTURES_REQUIRED root-tree-addresses-sueloader-fixture
                       FIXTURES_SETUP root-tree-addresses-ConfigRecord-fixture)

if(ClingWorkAroundMissingDynamicScope)
  SET(_sue_opts -e "(void)gROOT->ProcessLine(\".L sueloader.C+\")"
                -e "(void)gROOT->ProcessLine(\".L ConfigRecord.cxx+\")")
endif()

ROOTTEST_ADD_TEST(configtest
                  MACRO suetestwrt.C
                  ROOTEXE_OPTS ${_sue_opts}
                  OUTREF configtest.ref
                  FIXTURES_REQUIRED root-tree-addresses-ConfigRecord-fixture
                                    root-tree-addresses-sueloader-fixture
                 FIXTURES_SETUP root-tree-addresses-configtest-fixture)

ROOTTEST_ADD_TEST(suetest
                  MACRO suetestrd.C
                  ROOTEXE_OPTS ${_sue_opts}
                  OUTREF suetest.ref
                  FIXTURES_REQUIRED root-tree-addresses-ConfigRecord-fixture
                                    root-tree-addresses-sueloader-fixture
                                    root-tree-addresses-configtest-fixture)

ROOTTEST_COMPILE_MACRO(JansEvent.C
                       FIXTURES_SETUP root-tree-addresses-JansEvent-fixture)

ROOTTEST_ADD_TEST(Jantest
                  MACRO runJantest.C
                  COPY_TO_BUILDDIR janbug.root
                  OUTREF Jantest.ref
                  FIXTURES_REQUIRED root-tree-addresses-JansEvent-fixture)

ROOTTEST_ADD_TEST(deepnest
                  MACRO rundeepnest.C+
                  OUTREF deepnest.ref)

ROOTTEST_ADD_TEST(baseString
                  MACRO runbaseString.C+
                  OUTREF baseString.ref)

ROOTTEST_ADD_TEST(prova
                  MACRO runprova.C
                  COPY_TO_BUILDDIR prova1.root prova2.root
                  OUTREF prova.ref)

ROOTTEST_ADD_TEST(augertest
                  MACRO runaugertest.C
                  COPY_TO_BUILDDIR data/S_1_1004_qgsjet_100_1.KGrec.root
                                   data/S_1_104_qgsjet_100_1.KGrec.root
                  OUTREF augertest.ref)

ROOTTEST_ADD_TEST(longlong
                  MACRO runlonglong.C+
                  OUTCNVCMD sed -e s:0x[0-9a-fA-F]*:0xRemoved:g
                  OUTREF longlong.ref)

ROOTTEST_ADD_TEST(memleak
                  MACRO runmemleak.C+
                  COPY_TO_BUILDDIR memleak.root
                  OUTREF memleak.ref)

ROOTTEST_ADD_TEST(Leaflist
                  MACRO execLeaflist.C
                  COPY_TO_BUILDDIR leaflist.root
                  OUTREF execLeaflist.ref)

ROOTTEST_ADD_TEST(twrite
                  MACRO assert_twrite.cxx+
                  ERRREF assert_twrite.ref
                  FIXTURES_SETUP root-tree-addresses-twrite-fixture)

ROOTTEST_COMPILE_MACRO(assert_tread.cxx
                       FIXTURES_SETUP root-tree-addresses-assert_tread-fixture)

ROOTTEST_ADD_TEST(tread
                  MACRO assert_tread.cxx+
                  ERRREF assert_tread.ref
                  FIXTURES_REQUIRED root-tree-addresses-twrite-fixture root-tree-addresses-assert_tread-fixture)

ROOTTEST_ADD_TEST(ReuseTree
                  MACRO execReuseTree.C
                  OUTREF execReuseTree.ref)

ROOTTEST_ADD_TEST(BranchObjSplit0
                  MACRO execBranchObjSplit0.C
                  OUTREF execBranchObjSplit0.ref)

ROOTTEST_ADD_TEST(CheckErrors
                  MACRO execCheckErrors.C
                  OUTREF execCheckErrors.ref)

ROOTTEST_ADD_TEST(CMSobj
                  MACRO execCMSobj.cxx+
                  OUTREF execCMSobj.ref)

ROOTTEST_COMPILE_MACRO(jet.C
                       FIXTURES_SETUP root-tree-addresses-jet-fixture)

ROOTTEST_ADD_TEST(Jet
                  MACRO runJet.C
                  OUTREF Jet.ref
                  FIXTURES_REQUIRED root-tree-addresses-jet-fixture)