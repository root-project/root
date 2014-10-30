############################################################################
# CMakeLists.txt file for building ROOT tree/treeplayer package
# @author Pere Mato, CERN
############################################################################

set(libname TreePlayer)

ROOT_USE_PACKAGE(tree/tree)
ROOT_USE_PACKAGE(gui/gui)
ROOT_USE_PACKAGE(graf3d/g3d)


ROOT_GENERATE_DICTIONARY(G__${libname} *.h MODULE ${libname} LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")


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

