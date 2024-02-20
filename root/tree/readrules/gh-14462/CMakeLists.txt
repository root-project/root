################################################################################
# Regression test for https://github.com/root-project/root/issues/14462
################################################################################

ROOT_ADD_GTEST(gh14462regression gh14462regression.cxx LIBRARIES ${ROOT_LIBRARIES})
target_include_directories(gh14462regression PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
ROOT_GENERATE_DICTIONARY(gh14462regressionDict ${CMAKE_CURRENT_SOURCE_DIR}/userClass.hxx MODULE gh14462regression LINKDEF userClassLinkDef.hxx)
