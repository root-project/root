ROOTTEST_GENERATE_DICTIONARY(libArrayRules_dictcint
                  arrayRules.h
                  LINKDEF arrayRulesLinkDef.h
                  NO_ROOTMAP NO_CXXMODULE)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(libArrayRules_dictrflx
                  arrayRules.h
                  SELECTION arrayRulesSelection.xml
                  LIBNAME libArrayRules_dictrflx
                  NO_ROOTMAP)

ROOTTEST_ADD_TEST(Conversions
                  MACRO execConversions.cxx+
                  OUTREF execConversions.ref
                  LABELS longtest)

ROOTTEST_ADD_TEST(WriteOrderClass
                  MACRO execWriteOrderClass.cxx+
                  OUTREF execWriteOrderClass.ref
                  LABELS longtest
                  FIXTURES_SETUP WriteOrderClass)

ROOTTEST_ADD_TEST(ReadOrderClass
                  MACRO execReadOrderClass.cxx+
                  OUTREF execReadOrderClass.ref
                  LABELS longtest
                  FIXTURES_REQUIRED WriteOrderClass)

ROOTTEST_ADD_TEST(BuildMyClass
                  MACRO ${CMAKE_SOURCE_DIR}/roottest/scripts/build.C
                  MACROARG "\"MyClass.cxx\""
                  COPY_TO_BUILDDIR MyClass.h MyClass.cxx MyClass_linkdef.h
                  FIXTURES_SETUP BuildMyClass-fixture)

ROOTTEST_ADD_TEST(BuildSomeClass
                  MACRO ${CMAKE_SOURCE_DIR}/roottest/scripts/build.C
                  MACROARG "\"SomeClass.cxx\""
                  COPY_TO_BUILDDIR SomeClass.cxx
                  FIXTURES_SETUP BuildSomeClass-fixture)

ROOTTEST_ADD_TEST(writeTree
                  MACRO writeTree.C
                  ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L MyClass.cxx+\")" -e "(void)gROOT->ProcessLine(\".L SomeClass.cxx+\")"
                  OUTREF writeTree.ref
                  LABELS longtest
                  FIXTURES_REQUIRED BuildMyClass-fixture BuildSomeClass-fixture
                  FIXTURES_SETUP writeTree)

ROOTTEST_ADD_TEST(postobj
                  MACRO runpostobj.C
                  ROOTEXE_OPTS -e "(void)gROOT->ProcessLine(\".L MyClass.cxx+\")" -e "(void)gROOT->ProcessLine(\".L SomeClass.cxx+\")"
                  OUTREF postobj.ref
                  LABELS longtest
                  FIXTURES_REQUIRED writeTree)
