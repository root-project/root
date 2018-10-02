############################################################################
# CMakeLists.txt file for building ROOT math/smatrix package
############################################################################

ROOT_GENERATE_DICTIONARY(G__Smatrix
    Math/SMatrix.h
    Math/SMatrixDfwd.h
    Math/SMatrixFfwd.h
    Math/SVector.h
  MODULE
    Smatrix
  LINKDEF
    LinkDef.h
  OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
    MathCore
)

ROOT_GENERATE_DICTIONARY(G__Smatrix32
    Math/SMatrix.h
    Math/SMatrixDfwd.h
    Math/SMatrixFfwd.h
    Math/SVector.h
  MULTIDICT
  MODULE
    Smatrix
  LINKDEF
    LinkDefD32.h
  OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
    MathCore
)

ROOT_LINKER_LIBRARY(Smatrix
    G__Smatrix.cxx
    G__Smatrix32.cxx
  LIBRARIES
    Core
    MathCore
)

ROOT_INSTALL_HEADERS()
