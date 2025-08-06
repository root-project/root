ROOTTEST_GENERATE_DICTIONARY(libTreeFormulaReferencesGeneration
                             Event.h
                             LINKDEF EventLinkDef.h
                             SOURCES Event.cxx
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-treeformula-references-dict-fixture)

ROOTTEST_GENERATE_EXECUTABLE(TreeFormulaReferencesGenerator
                             TreeFormulaReferencesGenerator.cxx
                             LIBRARIES ROOT::Core ROOT::RIO ROOT::Net ROOT::Tree ROOT::Hist ROOT::MathCore libTreeFormulaReferencesGenerationlibgen
                             FIXTURES_REQUIRED root-treeformula-references-dict-fixture
                             FIXTURES_SETUP root-treeformula-references-generator-fixture)

set(size 6)
set(comp 0)
set(action 1)
set(tracks 30)

function(TREEFORMULA_REFERENCES_GENERATION name opt)
  ROOTTEST_ADD_TEST("${name}"
    COMMAND ./TreeFormulaReferencesGenerator ${size} ${comp} "${opt}" ${action} ${tracks} "TreeFormulaReferences.${name}.root"
    FIXTURES_REQUIRED root-treeformula-references-generator-fixture
    FIXTURES_SETUP root-treeformula-references-generation-fixture)
endfunction()

TREEFORMULA_REFERENCES_GENERATION(new.split0 0)
TREEFORMULA_REFERENCES_GENERATION(new.split1 1)
TREEFORMULA_REFERENCES_GENERATION(new.split2 2)
TREEFORMULA_REFERENCES_GENERATION(new.split9 9)
TREEFORMULA_REFERENCES_GENERATION(old.streamed -1)
TREEFORMULA_REFERENCES_GENERATION(old.split -2)

ROOTTEST_ADD_TEST(proxy
                 MACRO dt_RefProxyTest.C
                 COPY_TO_BUILDDIR dt_RefProxyTest.C
                 PASSRC 1
                 OUTREF RefProxyTest.ref
                 FIXTURES_REQUIRED root-treeformula-references-generation-fixture)

ROOTTEST_ADD_TEST(clones
                  COPY_TO_BUILDDIR execRefClonesArrayTest.h
                  OUTREF execRefClonesArrayTest.ref
                  MACRO execRefClonesArrayTest.cxx+)
