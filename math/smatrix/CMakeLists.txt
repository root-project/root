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
                                          LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")

ROOT_GENERATE_DICTIONARY(G__${libname}32  Math/SMatrix.h     Math/SVector.h
                                          Math/SMatrixDfwd.h Math/SMatrixFfwd.h
                                          Math/SMatrix.h     Math/SVector.h
                                          Math/SMatrixDfwd.h Math/SMatrixFfwd.h
                                          MULTIDICT MODULE ${libname}
                                          LINKDEF LinkDefD32.h OPTIONS "-writeEmptyRootPCM")

ROOT_LINKER_LIBRARY(${libname} G__${libname}.cxx G__${libname}32.cxx LIBRARIES Core)
ROOT_INSTALL_HEADERS()

