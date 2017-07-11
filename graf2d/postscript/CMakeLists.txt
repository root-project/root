############################################################################
# CMakeLists.txt file for building ROOT graf2d/postcript package
# @author Pere Mato, CERN
############################################################################

ROOT_GENERATE_DICTIONARY(G__Postscript *.h MODULE Postscript LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM" DEPENDENCIES Graf)

ROOT_LINKER_LIBRARY(Postscript *.cxx G__Postscript.cxx LIBRARIES  ${ZLIB_LIBRARIES} mathtext DEPENDENCIES Graf)
ROOT_INSTALL_HEADERS()
