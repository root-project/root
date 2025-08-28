ROOTTEST_ADD_TEST(StringFormula
                  MACRO testString.C
                  OUTREF testStringFormula.ref)

ROOTTEST_ADD_TEST(BoolOpt
                  MACRO testBoolOpt.C
                  OUTREF testBoolOpt.ref)

ROOTTEST_ADD_TEST(expotest
                  MACRO runexpotest.C
                  OUTREF expotest.ref)

# test was failing before, but now seems to be mean value calculated correctly
ROOTTEST_ADD_TEST(NonFormulaMean
                  MACRO NonFormulaMean.C
                  OUTREF NonFormulaMean.ref)

ROOTTEST_ADD_TEST(BoolOrder
                  MACRO runBoolOrder.C
                  COPY_TO_BUILDDIR mksm.root sm.root
                  OUTREF BoolOrder.ref)

if(NOT MSVC AND NOT ClingWorkAroundUnloadingVTABLES)
  # test uses grep internally therefore not run on Windows

  ROOTTEST_ADD_TEST(copymacros
                    COMMAND ${CMAKE_COMMAND} -E echo "copy macros"
                    COPY_TO_BUILDDIR forproxy.C templateName.cpp
                    FIXTURES_SETUP root-treeformula-parse-copy_macros-fixture)

  ROOTTEST_COMPILE_MACRO(templateName.cpp
                         FIXTURES_REQUIRED root-treeformula-parse-copy_macros-fixture
                         FIXTURES_SETUP root-treeformula-parse-templateName_cpp-fixture)

  # test has complex output which does not match on different platforms
  ROOTTEST_ADD_TEST(templateName
                    MACRO runtemplateName.C
                    OUTREF templateName.ref
                    ROOTEXE_OPTS -e "gROOT->SetMacroPath(\"${CMAKE_CURRENT_BINARY_DIR}\")"
                    OUTCNVCMD grep -v -e "ROOT version" -v -e "Creating library" -e "from Tree" -e "found on file" -e "TBranchProxy::Setup" -e "creating shared" -e "Warning in <TClassTable::Add>: class generatedSel already in TClassTable" -e "Not unloading file" -e "unloading" -e "::UnloadFile"
                    LABELS longtest
                    WILLFAIL
                    FIXTURES_REQUIRED root-treeformula-parse-templateName_cpp-fixture)
endif()

ROOTTEST_ADD_TEST(templateBase
                  MACRO runtemplateBase.C
                  COPY_TO_BUILDDIR CaloJets.root
                  OUTREF templateBase.ref)

# macro crashes on Windows, need to be investiaged
if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(aliases
                    MACRO runaliases.C
                    OUTREF aliases.ref)
endif()

ROOTTEST_ADD_TEST(AliasTest
                  MACRO execAliasTest.C
                  COPY_TO_BUILDDIR McGen-E5-T45-P0-MonteCarloMcHitsC100743.root
                  OUTREF execAliasTest.ref)

ROOTTEST_ADD_TEST(branchSlash
                  MACRO execbranchSlash.C
                  OUTREF execbranchSlash.ref)

ROOTTEST_COMPILE_MACRO(gausbug.cxx
                       FIXTURES_SETUP root-treeformula-parse-gausbug-fixture)

ROOTTEST_ADD_TEST(gaus
                  MACRO rungaus.C
                  OUTREF gaus.ref
                  FIXTURES_REQUIRED root-treeformula-parse-gausbug-fixture)

ROOTTEST_ADD_TEST(cscan
                  MACRO runcscan.C+
                  OUTREF cscan.ref)

ROOTTEST_COMPILE_MACRO(EventTcaMember.cc
                       FIXTURES_SETUP root-treeformula-parse-EventTcaMember-fixture)

ROOTTEST_ADD_TEST(functca
                  MACRO runfunctca.C
                  OUTREF functca.ref
                  ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L EventTcaMember.cc+\")"
                  FIXTURES_REQUIRED root-treeformula-parse-EventTcaMember-fixture)

ROOTTEST_COMPILE_MACRO(PEvent.cc
                       FIXTURES_SETUP root-treeformula-parse-PEvent-fixture)

ROOTTEST_ADD_TEST(PEvent
                  MACRO runPEvent.C
                  OUTREF PEvent.ref
                  ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L PEvent.cc+\")"
                  FIXTURES_REQUIRED root-treeformula-parse-PEvent-fixture)

# does not work, but not triggered in the makefile
ROOTTEST_ADD_TEST(strings
                  MACRO runstrings.C
                  COPY_TO_BUILDDIR itsdb.2006-03-17-22-28-46.root
                  WILLFAIL
                  OUTREF strings.ref)

ROOTTEST_COMPILE_MACRO(runbase.C
                       FIXTURES_SETUP root-treeformula-parse-runbase-fixture)

ROOTTEST_ADD_TEST(writebase
                  MACRO runbase.C+
                  MACROARG "true"
                  FIXTURES_REQUIRED root-treeformula-parse-runbase-fixture
                  FIXTURES_SETUP root-treeformula-parse-writebase-fixture)

ROOTTEST_ADD_TEST(base
                  MACRO runbase.C+
                  OUTREF base.ref
                  FIXTURES_REQUIRED root-treeformula-parse-writebase-fixture)

ROOTTEST_ADD_TEST(form
                  MACRO runform.C
                  OUTREF form.ref)

ROOTTEST_ADD_TEST(nodot
                  MACRO runnodot.C+
                  OUTREF nodot.ref)
