
ROOT_ADD_GTEST(testTBasket TBasket.cxx LIBRARIES RIO Tree)
ROOT_ADD_GTEST(testTBranch TBranch.cxx LIBRARIES RIO Tree MathCore)
ROOT_ADD_GTEST(testTIOFeatures TIOFeatures.cxx LIBRARIES RIO Tree)
if(imt)
   ROOT_ADD_GTEST(testTTreeImplicitMT ImplicitMT.cxx LIBRARIES RIO Tree)
endif()
