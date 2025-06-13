ROOTTEST_GENERATE_DICTIONARY(libRoottestIoEvolutionPragmaV1
                             v1/AliAODForwardMult.h
                             LINKDEF v1/AliAODForwardMultLinkDef.h
                             SOURCES v1/AliAODForwardMult.cxx
                             NO_CXXMODULE NO_ROOTMAP
                             FIXTURES_SETUP root-io-evolution-pragma-lib1)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(_nowarn -Wno-unused-variable)
endif()

ROOTTEST_GENERATE_DICTIONARY(libRoottestIoEvolutionPragmaV2
                             v2/AliAODForwardHeader.h v2/AliAODForwardMult.h
                             LINKDEF v2/AliAODForwardMultLinkDef.h
                             SOURCES v2/AliAODForwardHeader.cxx  v2/AliAODForwardMult.cxx
                             COMPILE_OPTIONS ${_nowarn}
                             NO_CXXMODULE NO_ROOTMAP
                             FIXTURES_SETUP root-io-evolution-pragma-lib2)

ROOTTEST_ADD_TEST(testv1
                  MACRO execTestv1.C
                  OUTREF execTestv1.ref
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-pragma-lib1
                  FIXTURES_SETUP root-io-evolution-pragma-testv1)

ROOTTEST_ADD_TEST(testv2
                  MACRO execTestv2.C
                  OUTREF execTestv2.ref
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-pragma-lib2 root-io-evolution-pragma-testv1)
