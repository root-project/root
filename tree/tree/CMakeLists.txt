############################################################################
# CMakeLists.txt file for building ROOT tree/tree package
# @author Pere Mato, CERN
############################################################################

include_directories(SYSTEM ${TBB_INCLUDE_DIRS})

ROOT_STANDARD_LIBRARY_PACKAGE(Tree
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              LIBRARIES ${TBB_LIBRARIES}
                              DEPENDENCIES Net RIO Thread Imt)

if(testing)
   add_subdirectory(test)
endif()
