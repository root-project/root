# LABELS longtest

ROOTTEST_ADD_TESTDIRS()

ROOTTEST_COMPILE_MACRO(cloningOne.C
                       FIXTURES_SETUP root-tree-evolution-cloningOne-fixture)

ROOTTEST_COMPILE_MACRO(cloningTwo.C
                       FIXTURES_SETUP root-tree-evolution-cloningTwo-fixture)

if(ClingWorkAroundMissingDynamicScope)
  set(_opts_cloningone -e "(void)gROOT->ProcessLine(\".L cloningOne.C+\")")
  set(_opts_cloningtwo -e "(void)gROOT->ProcessLine(\".L cloningTwo.C+\")")
endif()

ROOTTEST_ADD_TEST(cloningFile
                  MACRO runcloning.C
                  MACROARG 1
                  ROOTEXE_OPTS ${_opts_cloningone}
                  FIXTURES_REQUIRED root-tree-evolution-cloningOne-fixture root-tree-evolution-cloningTwo-fixture
                  FIXTURES_SETUP root-tree-evolution-cloningFile-fixture)

ROOTTEST_ADD_TEST(cloning
                  MACRO runcloning.C
                  ROOTEXE_OPTS ${_opts_cloningtwo}
                  FIXTURES_REQUIRED root-tree-evolution-cloningFile-fixture)

ROOTTEST_COMPILE_MACRO(MyClassClones.cxx
                       FIXTURES_SETUP root-tree-evolution-MyClassClones-fixture)

ROOTTEST_COMPILE_MACRO(MyClassVector.cxx
                       FIXTURES_SETUP root-tree-evolution-MyClassVector-fixture)

ROOTTEST_COMPILE_MACRO(MyClassList.cxx
                       FIXTURES_SETUP root-tree-evolution-MyClassList-fixture)

if(ClingWorkAroundMissingDynamicScope)
  set(_opts_clones -e "(void)gROOT->ProcessLine(\".L MyClassClones.cxx+\")")
  set(_opts_vector -e "(void)gROOT->ProcessLine(\".L MyClassVector.cxx+\")")
  set(_opts_list -e "(void)gROOT->ProcessLine(\".L MyClassList.cxx+\")")
endif()

ROOTTEST_ADD_TEST(WriteClones
                  MACRO WriteClones.C
                  ROOTEXE_OPTS ${_opts_clones}
                  FIXTURES_REQUIRED root-tree-evolution-MyClassClones-fixture
                  FIXTURES_SETUP root-tree-evolution-WriteClones-fixture)

ROOTTEST_ADD_TEST(WriteVector
                  MACRO WriteVector.C
                  ROOTEXE_OPTS ${_opts_vector}
                  FIXTURES_REQUIRED root-tree-evolution-MyClassVector-fixture
                  FIXTURES_SETUP root-tree-evolution-WriteVector-fixture)

ROOTTEST_ADD_TEST(WriteList
                  MACRO WriteList.C
                  ROOTEXE_OPTS ${_opts_list}
                  FIXTURES_REQUIRED root-tree-evolution-MyClassList-fixture
                  FIXTURES_SETUP root-tree-evolution-WriteList-fixture)

ROOTTEST_ADD_TEST(ReadClones
                  MACRO ReadClones.C
                  FIXTURES_REQUIRED root-tree-evolution-MyClassClones-fixture
                                    root-tree-evolution-WriteClones-fixture
                                    root-tree-evolution-WriteVector-fixture
                                    root-tree-evolution-WriteList-fixture
                  OUTREF ReadClones.ref)

ROOTTEST_ADD_TEST(ReadVector
                  MACRO ReadVector.C
                  FIXTURES_REQUIRED root-tree-evolution-MyClassVector-fixture
                                    root-tree-evolution-WriteClones-fixture
                                    root-tree-evolution-WriteVector-fixture
                                    root-tree-evolution-WriteList-fixture
                  OUTREF ReadVector.ref)

ROOTTEST_ADD_TEST(ReadList
                  MACRO ReadList.C
                  FIXTURES_REQUIRED root-tree-evolution-MyClassList-fixture
                                    root-tree-evolution-WriteClones-fixture
                                    root-tree-evolution-WriteVector-fixture
                                    root-tree-evolution-WriteList-fixture
                  OUTREF ReadList.ref)

ROOTTEST_ADD_TEST(ReadNolib
                  MACRO ReadNolib.C
                  FIXTURES_REQUIRED root-tree-evolution-WriteClones-fixture
                                    root-tree-evolution-WriteVector-fixture
                                    root-tree-evolution-WriteList-fixture
                  OUTREF ReadNolib.ref)

ROOTTEST_COMPILE_MACRO(SimpleOne.C
                       FIXTURES_SETUP root-tree-evolution-SimpleOne-fixture)

ROOTTEST_COMPILE_MACRO(SimpleTwo.C
                       FIXTURES_SETUP root-tree-evolution-SimpleTwo-fixture)

if(ClingWorkAroundMissingDynamicScope)
  set(_opts_one -e "(void)gROOT->ProcessLine(\".L SimpleOne.C+\")")
  set(_opts_two -e "(void)gROOT->ProcessLine(\".L SimpleTwo.C+\")")
endif()

ROOTTEST_ADD_TEST(WriteOne
                  MACRO SimpleWrite.C
                  MACROARG 1
                  ROOTEXE_OPTS ${_opts_one}
                  FIXTURES_REQUIRED root-tree-evolution-SimpleOne-fixture
                  FIXTURES_SETUP root-tree-evolution-WriteOne-fixture)

ROOTTEST_ADD_TEST(WriteTwo
                  MACRO SimpleWrite.C
                  MACROARG 2
                  ROOTEXE_OPTS ${_opts_two}
                  FIXTURES_REQUIRED root-tree-evolution-SimpleTwo-fixture
                  FIXTURES_SETUP root-tree-evolution-WriteTwo-fixture)

ROOTTEST_ADD_TEST(SimpleRead
                  MACRO SimpleRead.C
                  OUTREF SimpleRead.ref
                  FIXTURES_REQUIRED root-tree-evolution-WriteOne-fixture
                                    root-tree-evolution-WriteTwo-fixture)

ROOTTEST_ADD_TEST(writeMissingCollection
                  MACRO writeMissingCollection.cxx+
                  FIXTURES_SETUP root-tree-evolution-writeMissingCollection-fixture)

ROOTTEST_ADD_TEST(MissingCollection
                  MACRO execMissingCollection.cxx+
                  OUTREF execMissingCollection.ref
                  FIXTURES_REQUIRED root-tree-evolution-writeMissingCollection-fixture)

ROOTTEST_ADD_TEST(missingArray1
                  MACRO missingArray1.cc+
                  FIXTURES_SETUP root-tree-evolution-missingArray1-fixture)

ROOTTEST_ADD_TEST(missingArray2
                  MACRO missingArray2.cc+
                  OUTREF missingArray.ref
                  FIXTURES_REQUIRED root-tree-evolution-missingArray1-fixture)

