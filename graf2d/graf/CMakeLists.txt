############################################################################
# CMakeLists.txt file for building ROOT graf2d/graf package
# @author Pere Mato, CERN
############################################################################

set(libname Graf)

ROOT_USE_PACKAGE(core)
ROOT_USE_PACKAGE(math/matrix)
ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(graf2d/mathtext)
include_directories(${CMAKE_SOURCE_DIR}/hist/hist/inc)  # This is to avoid a circular dependency graf <--> hist

ROOT_GENERATE_DICTIONARY(G__${libname} *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(${libname} LINKDEF LinkDef.h DEPENDENCIES Hist Matrix MathCore RIO)

include_directories(${FREETYPE_INCLUDE_DIRS})

ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx LIBRARIES ${FREETYPE_LIBRARIES} ${ZLIB_LIB} mathtext DEPENDENCIES Hist Matrix MathCore RIO)

if(builtin_freetype)
  add_dependencies(${libname} FREETYPE)
endif()
ROOT_INSTALL_HEADERS()
