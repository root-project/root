############################################################################
# CMakeLists.txt file for building ROOT math/smatrix package
############################################################################

set(libname Smatrix)

ROOT_USE_PACKAGE(math)

ROOT_GENERATE_DICTIONARY(G__${libname}    Math/SMatrix.h     Math/SVector.h 
                                          Math/SMatrixDfwd.h Math/SMatrixFfwd.h
                                          Math/SMatrix.h     Math/SVector.h
                                          Math/SMatrixDfwd.h Math/SMatrixFfwd.h
                                          MODULE ${libname}
                                          LINKDEF LinkDefAll.h)

ROOT_LINKER_LIBRARY(${libname} G__${libname}.cxx LIBRARIES Core)
ROOT_INSTALL_HEADERS()

