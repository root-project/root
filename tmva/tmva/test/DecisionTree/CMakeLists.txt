############################################################################
# CMakeLists.txt file for building TMVA/DecisionTree tests.
# @author Sergey Divakov
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

set(Libraries Core MathCore Matrix TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

#--- Multi-target Regression tests. ------------------

add_library(DTreeUtils Utility.cpp)

include_directories(${TBB_INCLUDE_DIRS})

ROOT_ADD_GTEST(TestMultiRegression TestMultiregression.cpp LIBRARIES ${Libraries})
ROOT_ADD_GTEST(TestMultiVariance TestMultidimensionalVariance.cpp LIBRARIES ${Libraries})

target_link_libraries(TestMultiRegression DTreeUtils)
target_link_libraries(TestMultiVariance DTreeUtils)