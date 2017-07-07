############################################################################
# CMakeLists.txt file for building ROOT TMVA tests.
# @author Simon Pfreundschuh
############################################################################

project(tmva-tests)
find_package(ROOT REQUIRED)

ROOT_ADD_TEST_SUBDIRECTORY(DNN)
ROOT_ADD_TEST_SUBDIRECTORY(DNN/CNN)
