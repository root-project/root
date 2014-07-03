############################################################################
# CMakeLists.txt file for building ROOT hist/hbook package
############################################################################

set(libname Hbook)

ROOT_USE_PACKAGE(hist/hist)
ROOT_USE_PACKAGE(tree/treeplayer)

if(MSVC)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /SAFESEH:NO ")
  link_directories($ENV{LIB})
endif()
ROOT_GENERATE_DICTIONARY(G__${libname} *.h  LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(${libname} LINKDEF LinkDef.h DEPENDENCIES Hist Matrix Tree Graf TreePlayer RIO minicern )
ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx DEPENDENCIES Hist Matrix Tree Graf TreePlayer RIO minicern )
ROOT_INSTALL_HEADERS()

