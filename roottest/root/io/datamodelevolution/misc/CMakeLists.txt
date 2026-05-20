if(NOT MSVC)
  set(warnopt -Wno-unused-variable)
endif()

ROOTTEST_GENERATE_DICTIONARY(libArrayRules_dictcint
                  arrayRules.h
                  LINKDEF arrayRulesLinkDef.h
                  COMPILE_OPTIONS ${warnopt}
                  NO_ROOTMAP NO_CXXMODULE)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(libArrayRules_dictrflx
                  arrayRules.h
                  SELECTION arrayRulesSelection.xml
                  LIBNAME libArrayRules_dictrflx
                  COMPILE_OPTIONS ${warnopt}
                  NO_ROOTMAP)

ROOTTEST_ADD_TEST(Conversions
                  MACRO execConversions.cxx+
                  OUTREF execConversions.ref
                  LABELS longtest)

ROOTTEST_ADD_TEST(WriteOrderClass
                  MACRO execWriteOrderClass.cxx+
                  OUTREF execWriteOrderClass.ref
                  LABELS longtest
                  FIXTURES_SETUP root-io-datamodelevolution-misc-WriteOrderClass-fixture)

ROOTTEST_ADD_TEST(ReadOrderClass
                  MACRO execReadOrderClass.cxx+
                  OUTREF execReadOrderClass.ref
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-datamodelevolution-misc-WriteOrderClass-fixture)

ROOTTEST_COMPILE_MACRO(MyClass.cxx
                  FIXTURES_SETUP root-io-datamodelevolution-misc-BuildMyClass-fixture)

ROOTTEST_COMPILE_MACRO(SomeClass.cxx
                  FIXTURES_SETUP root-io-datamodelevolution-misc-BuildSomeClass-fixture)

if(ClingWorkAroundMissingDynamicScope)
  set(_myclass_load -e "(void)gROOT->ProcessLine(\".L MyClass.cxx+\")")
  set(_someclass_load -e "(void)gROOT->ProcessLine(\".L SomeClass.cxx+\")")
endif()

ROOTTEST_ADD_TEST(writeTree
                  MACRO writeTree.C
                  ROOTEXE_OPTS ${_myclass_load} ${_someclass_load}
                  OUTREF writeTree.ref
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-datamodelevolution-misc-BuildMyClass-fixture root-io-datamodelevolution-misc-BuildSomeClass-fixture
                  FIXTURES_SETUP root-io-datamodelevolution-misc-writeTree-fixture)

ROOTTEST_ADD_TEST(postobj
                  MACRO runpostobj.C
                  ROOTEXE_OPTS ${_myclass_load}
                  OUTREF postobj.ref
                  LABELS longtest
                  FIXTURES_REQUIRED root-io-datamodelevolution-misc-writeTree-fixture)
