############################################################################
# CMakeLists.txt file for building ROOT graf2d/graf package
# @author Pere Mato, CERN
############################################################################

ROOT_GENERATE_DICTIONARY(G__Graf MODULE Graf *.h LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")


include_directories(${FREETYPE_INCLUDE_DIRS})

ROOT_LINKER_LIBRARY(Graf *.cxx G__Graf.cxx LIBRARIES ${FREETYPE_LIBRARIES} ${ZLIB_LIBRARIES} mathtext DEPENDENCIES Hist Matrix MathCore RIO BUILTINS FREETYPE)

ROOT_INSTALL_HEADERS()
