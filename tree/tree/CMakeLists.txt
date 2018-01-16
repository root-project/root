############################################################################
# CMakeLists.txt file for building ROOT tree/tree package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Tree
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              LIBRARIES ${TBB_LIBRARIES}
                              DEPENDENCIES Net RIO Thread Imt)

ROOT_ADD_TEST_SUBDIRECTORY(test)
