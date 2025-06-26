ROOTTEST_GENERATE_DICTIONARY(libTreeFormulaScemaEvolution
                             Event.h
                             LINKDEF EventLinkDef.h
                             SOURCES Event.cxx
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-treeformula-schemaEvolution-lib1-fixture)

ROOTTEST_GENERATE_DICTIONARY(libTreeFormulaScemaEvolution2
                             Event_2.h
                             LINKDEF EventLinkDef.h
                             SOURCES Event_2.cxx
                             NO_ROOTMAP
                             NO_CXXMODULE
                             FIXTURES_SETUP root-treeformula-schemaEvolution-lib2-fixture)

ROOTTEST_GENERATE_EXECUTABLE(ScemaEvolutionMainEvent
                             MainEvent.cxx
                             LIBRARIES Core RIO libTreeFormulaScemaEvolutionlibgen
                             FIXTURES_REQUIRED root-treeformula-schemaEvolution-lib1-fixture
                             FIXTURES_SETUP root-treeformula-schemaEvolution-exec-fixture)

# write data with first version of event
ROOTTEST_ADD_TEST(write
                  COMMAND ./ScemaEvolutionMainEvent 10 0 7 1
                  FIXTURES_REQUIRED root-treeformula-schemaEvolution-exec-fixture
                  FIXTURES_SETUP root-treeformula-schemaEvolution-write-fixture)

# run all macros to produce output
foreach(arg 1 2 2a 3 3a 4 4a)
  ROOTTEST_ADD_TEST(schemaRun${arg}
                    MACRO schemaRun${arg}.C
                    OUTCNVCMD grep -v -e "no dictionary for class" -e Processing -e TRefArray -e TH1F
                    FIXTURES_REQUIRED root-treeformula-schemaEvolution-write-fixture
                                      root-treeformula-schemaEvolution-lib1-fixture
                                      root-treeformula-schemaEvolution-lib2-fixture
                    FIXTURES_SETUP root-treeformula-schemaEvolution-schemaRun${arg}-fixture)
endforeach()

# disable tests on Windows while sed is not available there
if(NOT MSVC OR win_broken_tests)
  # use first run as reference
  ROOTTEST_ADD_TEST(reference
                    COMMAND sed -e s,33554432,50331648,g -e s,33554456,50331672,g -e s,0x[0-9a-f]*,0xRemoved,g schemaRun1.log
                    WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}
                    STOREOUT
                    FIXTURES_REQUIRED root-treeformula-schemaEvolution-schemaRun1-fixture
                    FIXTURES_SETUP root-treeformula-schemaEvolution-reference-fixture)

  # compare all other with reference files
  foreach(arg 2 2a 3 3a 4 4a)
    ROOTTEST_ADD_TEST(schemaRun${arg}check
                      COMMAND sed -e s,33554432,50331648,g -e s,33554456,50331672,g -e s,0x[0-9a-f]*,0xRemoved,g schemaRun${arg}.log
                      WORKING_DIR ${CMAKE_CURRENT_BINARY_DIR}
                      OUTREF ${CMAKE_CURRENT_BINARY_DIR}/reference.log
                      FIXTURES_REQUIRED root-treeformula-schemaEvolution-schemaRun${arg}-fixture
                                        root-treeformula-schemaEvolution-reference-fixture)
  endforeach()
endif()
