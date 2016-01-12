############################################################################
# CMakeLists.txt file for building ROOT tree/treeplayer package
# @author Pere Mato, CERN
############################################################################

set(libname TreePlayer)

# TBranchProxyTemplate.h is only used by selectors, to verify that the selector
# source matches the ROOT interface. It should not end up in the dictionary nor
# in the PCH.
ROOT_GLOB_HEADERS(dictHeaders inc/*.h)
list(REMOVE_ITEM dictHeaders ${CMAKE_SOURCE_DIR}/tree/treeplayer/inc/TBranchProxyTemplate.h)

ROOT_GENERATE_DICTIONARY(G__${libname} ${dictHeaders} MODULE ${libname} LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")


ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx DEPENDENCIES Tree Graf3d Graf Hist Gpad RIO MathCore)
ROOT_INSTALL_HEADERS()


#---Extra rules-------------------------------------------------------
if(APPLE)
  if(GCC_MAJOR EQUAL 4 AND GCC_MINOR EQUAL 0 AND GCC_PATCH EQUAL 1)
    set_source_files_properties(src/TTreeFormula.cxx COMPILE_FLAGS "-O0" )
  endif()
  if(ICC_GE_9  EQUAL 10)
    set_source_files_properties(src/TTreeFormula.cxx COMPILE_FLAGS "-O0" )
  endif()
endif()

