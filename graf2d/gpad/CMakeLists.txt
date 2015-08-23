############################################################################
# CMakeLists.txt file for building ROOT graf2d/gpad package
# @author Pere Mato, CERN
############################################################################

include_directories(${CMAKE_SOURCE_DIR}/hist/hist/inc)  # This is to avoid a circular dependency graf <--> hist

ROOT_GENERATE_DICTIONARY(G__Gpad *.h MODULE Gpad LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")

ROOT_LINKER_LIBRARY(Gpad *.cxx G__Gpad.cxx DEPENDENCIES Graf Hist)
ROOT_INSTALL_HEADERS()
