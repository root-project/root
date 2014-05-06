############################################################################
# CMakeLists.txt file for building ROOT math/cmaes package
############################################################################

ROOT_USE_PACKAGE(math/mathcore)
ROOT_USE_PACKAGE(hist/hist)

add_definitions(-DWARNINGMSG -DUSE_ROOT_ERROR)

ROOT_GENERATE_DICTIONARY(G__cmaes *.h  cmaes/*.h MODULE cmaes LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(cmaes *.cxx G__cmaes.cxx DEPENDENCIES MathCore Hist)
ROOT_INSTALL_HEADERS()