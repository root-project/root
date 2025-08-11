include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(btag
                                    inc/BTagVertex_v1.h
                                    SELECTION inc/btag_selection.xml
                                    FIXTURES_SETUP root-meta-7181-btag-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(jet
                                    inc/Jet_v1.h
                                    SELECTION inc/jet_selection.xml
                                    FIXTURES_SETUP root-meta-7181-jet-fixture)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(sjet
                                    inc/SJet_v1.h
                                    SELECTION inc/sjet_selection.xml
                                    FIXTURES_SETUP root-meta-7181-sjet-fixture)

macro(ROOT_7181_TEST name use_rmap)
  if(${use_rmap} STREQUAL "true")
    set(testname rmap-${name})
  else()
    set(testname ${name})
  endif()

  ROOTTEST_ADD_TEST(${testname}
                    MACRO runscript.C
                    MACROARG "\"${CMAKE_CURRENT_SOURCE_DIR}/scripts/${name}.script\",${use_rmap}"
                    ROOTEXE_OPTS -e "gInterpreter->AddIncludePath(\"-I${CMAKE_CURRENT_SOURCE_DIR}\")"
                                 -e "gInterpreter->AddIncludePath(\"-I${CMAKE_CURRENT_SOURCE_DIR}/inc\")"
                    OUTREF ${testname}.ref
                    FIXTURES_REQUIRED root-meta-7181-btag-fixture
                                      root-meta-7181-jet-fixture
                                      root-meta-7181-sjet-fixture)
endmacro(ROOT_7181_TEST)

ROOT_7181_TEST(case1 false)
ROOT_7181_TEST(case2 false)
ROOT_7181_TEST(case3 false)
ROOT_7181_TEST(case4 false)

ROOT_7181_TEST(case1 true)
ROOT_7181_TEST(case2 true)
ROOT_7181_TEST(case3 true)
ROOT_7181_TEST(case4 true)
