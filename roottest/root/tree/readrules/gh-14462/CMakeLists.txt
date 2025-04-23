################################################################################
# Regression test for https://github.com/root-project/root/issues/14462
################################################################################

if(NOT (MSVC AND CMAKE_GENERATOR MATCHES Ninja))
  ROOT_GENERATE_DICTIONARY(gh14462regressionDict ${CMAKE_CURRENT_SOURCE_DIR}/userClass.hxx LINKDEF userClassLinkDef.hxx)
  ROOT_ADD_GTEST(gh14462regression gh14462regression.cxx gh14462regressionDict.cxx LIBRARIES ${ROOT_LIBRARIES})
  target_include_directories(gh14462regression PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()
