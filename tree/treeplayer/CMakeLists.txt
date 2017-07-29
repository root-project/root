############################################################################
# CMakeLists.txt file for building ROOT tree/treeplayer package
# @author Pere Mato, CERN
############################################################################

# TBranchProxyTemplate.h is only used by selectors, to verify that the selector
# source matches the ROOT interface. It should not end up in the dictionary nor
# in the PCH.
ROOT_GLOB_HEADERS(dictHeaders inc/*.h inc/ROOT/*.hxx)
list(REMOVE_ITEM dictHeaders ${CMAKE_SOURCE_DIR}/tree/treeplayer/inc/TBranchProxyTemplate.h)

ROOT_GLOB_SOURCES(sources src/*.cxx)

if(NOT imt)
  list(REMOVE_ITEM dictHeaders ${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT/TTreeProcessorMT.h)
  list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/TTreeProcessorMT.cxx)
else()
  include_directories(SYSTEM ${TBB_INCLUDE_DIRS})
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(TreePlayer
                              HEADERS ${dictHeaders}
                              SOURCES ${sources}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              LIBRARIES ${TBB_LIBRARIES}
                              DEPENDENCIES Tree Graf3d Graf Hist Gpad RIO MathCore MultiProc Imt)

#---Extra rules-------------------------------------------------------
if(APPLE)
  if(GCC_MAJOR EQUAL 4 AND GCC_MINOR EQUAL 0 AND GCC_PATCH EQUAL 1)
    set_source_files_properties(src/TTreeFormula.cxx COMPILE_FLAGS "-O0" )
  endif()
  if(ICC_GE_9  EQUAL 10)
    set_source_files_properties(src/TTreeFormula.cxx COMPILE_FLAGS "-O0" )
  endif()
endif()

if(testing)
  add_subdirectory(test)
endif()
