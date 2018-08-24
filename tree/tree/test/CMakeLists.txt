
ROOT_GENERATE_DICTIONARY(ElementStructDict ElementStruct.h LINKDEF ElementStructLinkDef.h OPTIONS -inlineInputHeader)
ROOT_ADD_GTEST(testTOffsetGeneration TOffsetGeneration.cxx ElementStruct.cxx ElementStructDict.cxx
  COPY_TO_BUILDDIR ElementStruct.h
  LIBRARIES RIO Tree MathCore
)
target_include_directories(testTOffsetGeneration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
ROOT_ADD_GTEST(testBulkApi BulkApi.cxx LIBRARIES RIO Tree TreePlayer)
ROOT_ADD_GTEST(testBulkApiMultiple BulkApiMultiple.cxx LIBRARIES RIO Tree TreePlayer)
ROOT_ADD_GTEST(testTBasket TBasket.cxx LIBRARIES RIO Tree)
ROOT_ADD_GTEST(testTBranch TBranch.cxx LIBRARIES RIO Tree MathCore)
ROOT_ADD_GTEST(testTIOFeatures TIOFeatures.cxx LIBRARIES RIO Tree)
ROOT_ADD_GTEST(testTTreeCluster TTreeClusterTest.cxx LIBRARIES RIO Tree MathCore)
if(imt)
   ROOT_ADD_GTEST(testTTreeImplicitMT ImplicitMT.cxx LIBRARIES RIO Tree)
endif()
ROOT_ADD_GTEST(testTChainSaveAsCxx TChainSaveAsCxx.cxx LIBRARIES RIO Tree)
ROOT_ADD_GTEST(testTTreeTruncatedDatatypes TTreeTruncatedDatatypes.cxx LIBRARIES RIO Tree)
