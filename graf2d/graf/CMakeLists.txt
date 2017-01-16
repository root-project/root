############################################################################
# CMakeLists.txt file for building ROOT graf2d/graf package
# @author Pere Mato, CERN
############################################################################

set(libname Graf)

ROOT_GENERATE_DICTIONARY(G__${libname} MODULE ${libname} *.h LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")


include_directories(${FREETYPE_INCLUDE_DIRS})

ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx LIBRARIES ${FREETYPE_LIBRARIES} ${ZLIB_LIBRARIES} mathtext DEPENDENCIES Hist Matrix MathCore RIO)

ROOT_INSTALL_HEADERS()
