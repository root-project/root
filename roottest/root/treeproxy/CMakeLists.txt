ROOTTEST_ADD_TESTDIRS()

ROOTTEST_ADD_TEST(leaflist
                  MACRO runleaflist.C
                  COPY_TO_BUILDDIR simPosProxy.C hsimpleProxy.C
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  OUTREF leaflist.ref)

ROOTTEST_ADD_TEST(createRed
                  MACRO createRed.C
                  COPY_TO_BUILDDIR reduced.N00008257_0002.spill.sntp.R1_18.0.root sum.C
                  FIXTURES_SETUP root-treeproxy-createRed-fixture)

ROOTTEST_ADD_TEST(reducedSpill
                  MACRO runreducedSpill.C
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  OUTREF reducedSpill.ref
                  FIXTURES_REQUIRED root-treeproxy-createRed-fixture
                  FIXTURES_SETUP root-treeproxy-reducedSpill-fixture)


# was not run in Makefile, run after reducedSpill while creating same red.h file again
ROOTTEST_ADD_TEST(producereducedSpill
                  MACRO producereducedSpill.C
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  OUTREF producereducedSpill.ref
                  FIXTURES_REQUIRED root-treeproxy-reducedSpill-fixture)

ROOTTEST_ADD_TEST(createSearch
                  MACRO createSearch.C
                  COPY_TO_BUILDDIR p0dNuESearch.root pns.C
                  FIXTURES_SETUP root-treeproxy-createSearch-fixture)

ROOTTEST_ADD_TEST(execsearch
                  MACRO execsearch.C
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  OUTREF execsearch.ref
                  FIXTURES_REQUIRED root-treeproxy-createSearch-fixture)

# copy files that produced TTrack class has proper set of class declaration file
ROOTTEST_ADD_TEST(copydim3files
                  COMMAND ${CMAKE_COMMAND} -E echo "copy dim3 files"
                  COPY_TO_BUILDDIR val3dim.C Track.h Track.C
                  FIXTURES_SETUP root-treeproxy-copydim3files-fixture)

ROOTTEST_COMPILE_MACRO(Track.C
                       FIXTURES_REQUIRED root-treeproxy-copydim3files-fixture
                       FIXTURES_SETUP root-treeproxy-Track-fixture)

ROOTTEST_COMPILE_MACRO(createvaldim3.C
                       BUILDLIB Track_C
                       FIXTURES_REQUIRED root-treeproxy-Track-fixture
                       FIXTURES_SETUP root-treeproxy-createvaldim3-fixture)

ROOTTEST_ADD_TEST(val3dimSel
                  MACRO createvaldim3.C+
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  FIXTURES_REQUIRED root-treeproxy-createvaldim3-fixture
                  FIXTURES_SETUP root-treeproxy-val3dimSel-fixture)

ROOTTEST_COMPILE_MACRO(${CMAKE_CURRENT_BINARY_DIR}/val3dimSel.h
                       FIXTURES_REQUIRED root-treeproxy-val3dimSel-fixture
                       FIXTURES_SETUP root-treeproxy-val3dimSel_compile-fixture)

ROOTTEST_COMPILE_MACRO(runvaldim3.C
                       FIXTURES_SETUP root-treeproxy-runvaldim3-fixture)

ROOTTEST_ADD_TEST(valdim3
                  MACRO runvaldim3.C+
                  OUTREF valdim3.ref
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                               -e "(void)gSystem->Load(\"Track_C\")"
                  FIXTURES_REQUIRED root-treeproxy-val3dimSel-fixture
                                    root-treeproxy-val3dimSel_compile-fixture
                                    root-treeproxy-Track-fixture
                                    root-treeproxy-runvaldim3-fixture)

ROOTTEST_COMPILE_MACRO(runfullmc.C
                       FIXTURES_SETUP root-treeproxy-runfullmc-fixture)
ROOTTEST_ADD_TEST(fullmcsel
                  MACRO runfullmc.C+
                  MACROARG 4
                  COPY_TO_BUILDDIR full-mcfile.root fullmc.cxx
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  FIXTURES_REQUIRED root-treeproxy-runfullmc-fixture
                  FIXTURES_SETUP root-treeproxy-fullmcsel-fixture)

ROOTTEST_ADD_TEST(fullmc
                  MACRO runfullmc.C+
                  MACROARG 5
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  OUTREF fullmc.ref
                  FIXTURES_REQUIRED root-treeproxy-runfullmc-fixture
                                    root-treeproxy-fullmcsel-fixture)

ROOTTEST_ADD_TEST(make_tuple
                  MACRO runmake_tuple.C
                  COPY_TO_BUILDDIR make_tuple.C make_tuple_draw.C
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  OUTREF make_tuple.ref)

ROOTTEST_ADD_TEST(friendchain
                  MACRO runfriendchain.C
                  COPY_TO_BUILDDIR vertex.C make_chains.cxx
                                   longtree1.root longtree2.root
                                   floattree1.root floattree2.root
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  OUTREF friendchain.ref)

ROOTTEST_ADD_TEST(btagobjsSel
                  MACRO runbtagobjs.C
                  MACROARG 4
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  COPY_TO_BUILDDIR btagobjs.root btagobjsScript.h btagobjsScript.cxx
                  FIXTURES_SETUP root-treeproxy-btagobjsSel-fixture)

ROOTTEST_ADD_TEST(btagobjsProxy
                  MACRO runbtagobjs.C
                  MACROARG 5
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  OUTREF btagobjsProxy.ref
                  FIXTURES_REQUIRED root-treeproxy-btagobjsSel-fixture)

ROOTTEST_ADD_TEST(VectorProxyTest
                  MACRO execVectorProxyTest.C
                  COPY_TO_BUILDDIR VectorProxyTest.h VectorProxyTest.C VectorProxyTest.root
                  ROOTEXE_OPTS -e "(void)gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                  OUTREF execVectorProxyTest.ref)
