ROOTTEST_ADD_TEST(mkdir
                  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/generated_selectors
                  FIXTURES_SETUP root-tree-selectorreader-mkdir-fixture)

ROOTTEST_COMPILE_MACRO(SampleClasses.h
                       FIXTURES_SETUP root-tree-selectorreader-SampleClasses-fixture)

ROOTTEST_ADD_TEST(Classes
                  MACRO runClasses.C
                  MACROARG "\"${CMAKE_CURRENT_SOURCE_DIR}\""
                  OUTREF Classes.ref
                  FIXTURES_REQUIRED root-tree-selectorreader-mkdir-fixture root-tree-selectorreader-SampleClasses-fixture)

ROOTTEST_ADD_TEST(CollectionClasses
                  MACRO runCollectionClasses.C
                  MACROARG "\"${CMAKE_CURRENT_SOURCE_DIR}\""
                  OUTREF CollectionClasses.ref
                  FIXTURES_REQUIRED root-tree-selectorreader-mkdir-fixture root-tree-selectorreader-SampleClasses-fixture)

ROOTTEST_ADD_TEST(Collections
                  MACRO runCollections.C
                  MACROARG "\"${CMAKE_CURRENT_SOURCE_DIR}\""
                  OUTREF Collections.ref
                  FIXTURES_REQUIRED root-tree-selectorreader-mkdir-fixture root-tree-selectorreader-SampleClasses-fixture)

ROOTTEST_ADD_TEST(Options
                  MACRO runOptions.C
                  MACROARG "\"${CMAKE_CURRENT_SOURCE_DIR}\""
                  OUTREF Options.ref
                  FIXTURES_REQUIRED root-tree-selectorreader-mkdir-fixture root-tree-selectorreader-SampleClasses-fixture)

ROOTTEST_ADD_TEST(Simple
                  MACRO runSimple.C
                  MACROARG "\"${CMAKE_CURRENT_SOURCE_DIR}\""
                  OUTREF Simple.ref
                  FIXTURES_REQUIRED root-tree-selectorreader-mkdir-fixture)


