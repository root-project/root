############################################################################
# CMakeLists.txt file for building ROOT TMVA tests.
# @author Kim Albertsson
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

set(Libraries TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

# Tests
ROOT_EXECUTABLE(TestRandomGenerator
                TestRandomGenerator.cxx
                LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-RandomGenerator
              COMMAND TestRandomGenerator)

# Subdirectories
ROOT_ADD_TEST_SUBDIRECTORY(test/crossvalidation)
ROOT_ADD_TEST_SUBDIRECTORY(test/DNN)
ROOT_ADD_TEST_SUBDIRECTORY(test/Method)
ROOT_ADD_TEST_SUBDIRECTORY(test/ROC)
ROOT_ADD_TEST_SUBDIRECTORY(test/envelope)