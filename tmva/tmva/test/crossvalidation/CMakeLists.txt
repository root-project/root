############################################################################
# CMakeLists.txt file for building ROOT TMVA tests.
# @author Kim Albertsson
############################################################################

set(Libraries TMVA)
include_directories(${ROOT_INCLUDE_DIRS})

# Tests utilising google test
ROOT_ADD_GTEST(testCrossValidationSplitting
               TestCrossValidationSplitting.cxx
               LIBRARIES ${Libraries})
ROOT_ADD_GTEST(testCrossValidationMultiProc
               TestCrossValidationMultiProc.cxx
               LIBRARIES ${Libraries})

# Tests
ROOT_EXECUTABLE(testCrossValidationSerialise
                TestCrossValidationSerialise.cxx
                LIBRARIES ${Libraries})
ROOT_ADD_TEST(TMVA-CrossValidation-Serialise
              COMMAND testCrossValidationSerialise)