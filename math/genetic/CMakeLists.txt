############################################################################
# CMakeLists.txt file for building ROOT math/genetic package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Genetic
                              HEADERS Math/GeneticMinimizer.h
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES Core MathCore TMVA)

ROOT_ADD_TEST_SUBDIRECTORY(test)
