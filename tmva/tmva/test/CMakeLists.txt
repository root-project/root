############################################################################
# CMakeLists.txt file for building ROOT TMVA tests.
# @author Simon Pfreundschuh
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

ROOT_ADD_TEST_SUBDIRECTORY(DNN)
ROOT_ADD_TEST_SUBDIRECTORY(DNN/RNN)

# for gtest
#ROOT_ADD_UNITTEST_DIR(TMVA)
#ROOT_ADD_UNITTEST_DIR(TMVA DNN)
#ROOT_ADD_UNITTEST_DIR(TMVA DNN RNN)
