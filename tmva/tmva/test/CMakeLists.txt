############################################################################
# CMakeLists.txt file for building ROOT TMVA tests.
# @author Kim Albertsson
############################################################################

set(Libraries TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

# Tests using google test
ROOT_ADD_GTEST(TestRandomGenerator
               TestRandomGenerator.cxx
               LIBRARIES ${Libraries})

if(dataframe)
    ROOT_ADD_GTEST(rtensor rtensor.cxx LIBRARIES ROOTVecOps TMVA)
    ROOT_ADD_GTEST(rtensor-utils rtensor_utils.cxx LIBRARIES ROOTVecOps TMVA ROOTDataFrame)
endif()

project(tmva-tests)
find_package(ROOT REQUIRED)


