############################################################################
# CMakeLists.txt file for building ROOT math/smatrix package
############################################################################

set(libname Smatrix)

ROOT_USE_PACKAGE(math)

ROOT_GENERATE_DICTIONARY(G__${libname}    Math/SMatrix.h     Math/SVector.h 
                                          Math/SMatrixDfwd.h Math/SMatrixFfwd.h LINKDEF LinkDef.h)
ROOT_GENERATE_DICTIONARY(G__${libname}32  Math/SMatrix.h     Math/SVector.h 
                                          Math/SMatrixDfwd.h Math/SMatrixFfwd.h LINKDEF LinkDefD32.h)

if(CXX_HAS_Wno-array-bounds)
  set_source_files_properties(G__${libname}.cxx G__${libname}32.cxx COMPILE_FLAGS "-Wno-array-bounds" )
endif()

ROOT_GENERATE_ROOTMAP(${libname} LINKDEF LinkDef.h )
ROOT_LINKER_LIBRARY(${libname} G__${libname}.cxx G__${libname}32.cxx LIBRARIES Core Cint)
ROOT_INSTALL_HEADERS()

