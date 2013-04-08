############################################################################
# CMakeLists.txt file for building ROOT io/io package
############################################################################

set(libname RIO)

ROOT_USE_PACKAGE(core)
ROOT_USE_PACKAGE(core/metautils)
ROOT_USE_PACKAGE(core/thread)
ROOT_USE_PACKAGE(math/mathcore)


ROOT_GENERATE_DICTIONARY(G__IO *.h  LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(${libname} LINKDEF LinkDef.h )

ROOT_LINKER_LIBRARY(${libname} *.cxx G__IO.cxx LIBRARIES ${CMAKE_DL_LIBS}
                                               DEPENDENCIES Core Thread)
ROOT_INSTALL_HEADERS()

#--- Extra rules ----------------------------------------------------------
set_source_files_properties( ${CMAKE_CURRENT_SOURCE_DIR}/src/TStreamerInfoReadBuffer.cxx 
                             COMPILE_FLAGS -DR__EXPLICIT_FUNCTION_INSTANTIATION)
