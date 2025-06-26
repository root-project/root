ROOT_GENERATE_DICTIONARY(G__TreeFormulaReferencesGeneration
  ${CMAKE_CURRENT_SOURCE_DIR}/Event.h
  LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/EventLinkDef.h)

ROOTTEST_LINKER_LIBRARY(TreeFormulaReferencesGeneration
  ${CMAKE_CURRENT_SOURCE_DIR}/TreeFormulaReferencesGeneration.cxx
  G__TreeFormulaReferencesGeneration.cxx
  LIBRARIES ROOT::Core ROOT::Tree ROOT::Hist ROOT::MathCore)

ROOTTEST_GENERATE_EXECUTABLE(TreeFormulaReferencesGenerator
  ${CMAKE_CURRENT_SOURCE_DIR}/TreeFormulaReferencesGenerator.cxx
  LIBRARIES ROOT::Core ROOT::RIO ROOT::Net ROOT::Tree ROOT::Hist ROOT::MathCore TreeFormulaReferencesGeneration
  DEPENDS TreeFormulaReferencesGeneration
  FIXTURES_SETUP treeformula-references-generator)

set(size 6)
set(comp 0)
set(action 1)
set(tracks 30)

function(TREEFORMULA_REFERENCES_GENERATION name opt)
  ROOTTEST_ADD_TEST("${name}"
    COMMAND ./TreeFormulaReferencesGenerator ${size} ${comp} "${opt}" ${action} ${tracks} "TreeFormulaReferences.${name}.root"
    FIXTURES_REQUIRED treeformula-references-generator
    FIXTURES_SETUP treeformula-references-generation
    RESOURCE_LOCK treeformula-references-file)
endfunction()

TREEFORMULA_REFERENCES_GENERATION(new.split0 0)
TREEFORMULA_REFERENCES_GENERATION(new.split1 1)
TREEFORMULA_REFERENCES_GENERATION(new.split2 2)
TREEFORMULA_REFERENCES_GENERATION(new.split9 9)
TREEFORMULA_REFERENCES_GENERATION(old.streamed -1)
TREEFORMULA_REFERENCES_GENERATION(old.split -2)

configure_file(dt_RefProxyTest.C . COPYONLY)

ROOTTEST_ADD_TEST(proxy
  OUTREF RefProxyTest.ref
  MACRO dt_RefProxyTest.C
  PASSRC 1
  FIXTURES_REQUIRED treeformula-references-generation)

ROOTTEST_ADD_TEST(clones
  COPY_TO_BUILDDIR execRefClonesArrayTest.h
  OUTREF execRefClonesArrayTest.ref
  MACRO  execRefClonesArrayTest.cxx)
