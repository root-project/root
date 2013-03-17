############################################################################
# CMakeLists.txt file for building ROOT graf2d/x11 package
# @author Pere Mato, CERN
############################################################################
set(libname GX11)

ROOT_USE_PACKAGE(core)
include_directories(${X11_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__${libname} T*.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(${libname} LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(${libname} *.cxx  *.c G__${libname}.cxx LIBRARIES Core ${X11_LIBRARIES} ${X11_Xpm_LIB} ${CMAKE_THREAD_LIBS_INIT})
ROOT_INSTALL_HEADERS()
