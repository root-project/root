############################################################################
# CMakeLists.txt file for building ROOT io/io package
############################################################################

set(libname RIO)

ROOT_USE_PACKAGE(core)
ROOT_USE_PACKAGE(core/metautils)
ROOT_USE_PACKAGE(core/thread)
ROOT_USE_PACKAGE(math/mathcore)


ROOT_GENERATE_DICTIONARY(G__IO *.h STAGE1 MODULE ${libname} LINKDEF LinkDef.h)

ROOT_OBJECT_LIBRARY(RIOObjs G__IO.cxx  *.cxx)
ROOT_LINKER_LIBRARY(${libname} $<TARGET_OBJECTS:RIOObjs>
                               LIBRARIES ${CMAKE_DL_LIBS}
                               DEPENDENCIES Core Thread)
ROOT_INSTALL_HEADERS()

#--- Extra rules ----------------------------------------------------------
