add_library(ROOTUnitTestSupport OBJECT ROOTUnitTestSupport.cxx)
target_include_directories(ROOTUnitTestSupport INTERFACE .)
target_link_libraries(ROOTUnitTestSupport PUBLIC Core gtest)

