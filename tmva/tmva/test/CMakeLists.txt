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

# Subdirectories
ROOT_ADD_TEST_SUBDIRECTORY(test/crossvalidation)
ROOT_ADD_TEST_SUBDIRECTORY(test/DNN)
ROOT_ADD_TEST_SUBDIRECTORY(test/Method)
ROOT_ADD_TEST_SUBDIRECTORY(test/ROC)
ROOT_ADD_TEST_SUBDIRECTORY(test/envelope)