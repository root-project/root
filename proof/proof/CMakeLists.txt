############################################################################
# CMakeLists.txt file for building ROOT proof/proof package
# @author Pere Mato, CERN
############################################################################

ROOT_GLOB_HEADERS(headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inc inc/*.h)
ROOT_GLOB_SOURCES(sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src src/*.cxx)

#--- AliEn Dataset manager only if AliEn is available
if(alien)
  add_definitions(-DALIENDSMGR)
else()
  list(REMOVE_ITEM headers TDataSetManagerAliEn.h)
  list(REMOVE_ITEM sources TDataSetManagerAliEn.cxx)
endif()

ROOT_GENERATE_DICTIONARY(G__Proof ${headers} MODULE Proof LINKDEF LinkDef.h DEPENDENCIES Net Tree Thread RIO MathCore)

ROOT_LINKER_LIBRARY(Proof ${sources} G__Proof.cxx DEPENDENCIES Net Tree Thread RIO MathCore)
ROOT_INSTALL_HEADERS()


