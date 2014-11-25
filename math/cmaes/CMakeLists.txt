############################################################################
# CMakeLists.txt file for building ROOT math/cmaes package
############################################################################

ROOT_USE_PACKAGE(math/mathcore)
ROOT_USE_PACKAGE(hist/hist)

add_definitions(-DWARNINGMSG -DUSE_ROOT_ERROR)

include_directories(${LIBCMAES_INCLUDE_DIR})
link_directories(${LIBCMAES_LIBRARIES})

ROOT_GENERATE_DICTIONARY(G__cmaes *.h MODULE cmaes LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(cmaes_root *.cxx G__cmaes.cxx DEPENDENCIES MathCore Hist LIBRARIES cmaes)
ROOT_INSTALL_HEADERS()
