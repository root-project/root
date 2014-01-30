############################################################################
# CMakeLists.txt file for building ROOT montecarlo/eg package
# @author Pere Mato, CERN
############################################################################

set(libname EG)

ROOT_USE_PACKAGE(graf2d/gpad)
ROOT_USE_PACKAGE(graf3d/g3d)
ROOT_USE_PACKAGE(math/mathcore)
ROOT_USE_PACKAGE(math/physics)

ROOT_GENERATE_DICTIONARY(G__${libname} T*.h MODULE ${libname} LINKDEF LinkDef.h)


ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx DEPENDENCIES Graf3d Graf Gpad MathCore )
ROOT_INSTALL_HEADERS()
