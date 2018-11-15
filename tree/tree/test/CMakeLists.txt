
ROOT_STANDARD_LIBRARY_PACKAGE(ElementStruct NO_INSTALL_HEADERS HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/ElementStruct.h SOURCES ElementStruct.cxx LINKDEF ElementStructLinkDef.h DEPENDENCIES RIO)
ROOT_ADD_GTEST(testTOffsetGeneration TOffsetGeneration.cxx LIBRARIES RIO Tree MathCore ElementStruct)
ROOT_ADD_GTEST(testTBasket TBasket.cxx LIBRARIES RIO Tree)
ROOT_ADD_GTEST(testTBranch TBranch.cxx LIBRARIES RIO Tree MathCore)
ROOT_ADD_GTEST(testTIOFeatures TIOFeatures.cxx LIBRARIES RIO Tree)
ROOT_ADD_GTEST(testTTreeCluster TTreeClusterTest.cxx LIBRARIES RIO Tree MathCore)
if(imt)
   ROOT_ADD_GTEST(testTTreeImplicitMT ImplicitMT.cxx LIBRARIES RIO Tree)
endif()

