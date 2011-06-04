############################################################################
# CMakeLists.txt file for building ROOT graf2d/postcript package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(graf2d/graf)

ROOT_GENERATE_DICTIONARY(G__PostScript *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Postscript LINKDEF LinkDef.h DEPENDENCIES Graf)
ROOT_LINKER_LIBRARY(Postscript *.cxx G__PostScript.cxx DEPENDENCIES Graf)
ROOT_INSTALL_HEADERS()
