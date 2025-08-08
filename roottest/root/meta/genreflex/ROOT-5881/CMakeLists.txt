
ROOTTEST_GENERATE_REFLEX_DICTIONARY(StringDict
                                    stringIO_allClasses.h
                                    SELECTION stringIO_allClasses_selection.xml
                                    FIXTURES_SETUP root-meta-genreflex-5881-dict-fixture)

ROOTTEST_GENERATE_EXECUTABLE(writeTreeBinary
                             writeTree.cpp
                             LIBRARIES Core RIO Tree
                             FIXTURES_SETUP root-meta-genreflex-5881-exec-fixture)

ROOTTEST_ADD_TEST(execStringInputFiles
                  EXEC ${CMAKE_CURRENT_BINARY_DIR}/writeTreeBinary
                  FIXTURES_REQUIRED root-meta-genreflex-5881-dict-fixture
                                    root-meta-genreflex-5881-exec-fixture
                  FIXTURES_SETUP root-meta-genreflex-5881-inputfiles-fixture)

ROOTTEST_ADD_TEST(execstringI
                  MACRO execstringI.C
                  OUTREF execstringI.ref
                  FIXTURES_REQUIRED root-meta-genreflex-5881-inputfiles-fixture)
