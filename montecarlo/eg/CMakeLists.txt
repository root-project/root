############################################################################
# CMakeLists.txt file for building ROOT montecarlo/eg package
# @author Pere Mato, CERN
############################################################################

set(libname EG)

ROOT_GENERATE_DICTIONARY(G__${libname} T*.h MODULE ${libname} LINKDEF LinkDef.h DEPENDENCIES Graf3d Graf Gpad MathCore )


ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx DEPENDENCIES Graf3d Graf Gpad MathCore )
ROOT_INSTALL_HEADERS()
