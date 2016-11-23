############################################################################
# CMakeLists.txt file for building ROOT graf2d/postcript package
# @author Pere Mato, CERN
############################################################################

ROOT_GENERATE_DICTIONARY(G__PostScript *.h MODULE Postscript LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")

ROOT_LINKER_LIBRARY(Postscript *.cxx G__PostScript.cxx LIBRARIES  ${ZLIB_LIBRARIES} mathtext DEPENDENCIES Graf)
ROOT_INSTALL_HEADERS()
