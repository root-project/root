
ROOT_STANDARD_LIBRARY_PACKAGE(ElementStruct NO_INSTALL_HEADERS HEADERS ElementStruct.h SOURCES ElementStruct.cxx LINKDEF ElementStructLinkDef.h DEPENDENCIES RIO)
if(MSVC)
  add_custom_command(TARGET ElementStruct POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/libElementStruct.dll
                                     ${CMAKE_CURRENT_BINARY_DIR}/libElementStruct.dll)
endif()
ROOT_ADD_GTEST(testTOffsetGeneration TOffsetGeneration.cxx LIBRARIES RIO Tree MathCore ElementStruct)
ROOT_ADD_GTEST(testTBasket TBasket.cxx LIBRARIES RIO Tree)
ROOT_ADD_GTEST(testTBranch TBranch.cxx LIBRARIES RIO Tree MathCore)
ROOT_ADD_GTEST(testTIOFeatures TIOFeatures.cxx LIBRARIES RIO Tree)
ROOT_ADD_GTEST(testTTreeCluster TTreeClusterTest.cxx LIBRARIES RIO Tree MathCore)
if(imt)
   ROOT_ADD_GTEST(testTTreeImplicitMT ImplicitMT.cxx LIBRARIES RIO Tree)
endif()
ROOT_ADD_GTEST(testTChainSaveAsCxx TChainSaveAsCxx.cxx LIBRARIES RIO Tree)
ROOT_ADD_GTEST(testTTreeTruncatedDatatypes TTreeTruncatedDatatypes.cxx LIBRARIES RIO Tree)
