############################################################################
# CMakeLists.txt file for building ROOT gui/fitpanelv7
############################################################################

set(libname ROOTFitPanel7)

ROOT_GLOB_HEADERS(ROOTFitPanel7_headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT/*.hxx)
ROOT_GLOB_SOURCES(ROOTFitPanel7_sources src/*.cxx)

ROOT_GENERATE_DICTIONARY(G__${libname} ${ROOTFitPanel7_headers} MODULE ${libname} LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef.h OPTIONS DEPENDENCIES ROOTWebDisplay ROOTHistDraw ROOTGpadv7)

ROOT_OBJECT_LIBRARY(ROOTFitPanel7Objs G__ROOTFitPanel7.cxx ${ROOTFitPanel7_sources})

ROOT_LINKER_LIBRARY(${libname} $<TARGET_OBJECTS:ROOTFitPanel7Objs> $<TARGET_OBJECTS:RootPcmObjs>
                    DEPENDENCIES ROOTWebDisplay ROOTHistDraw ROOTGpadv7)

ROOT_INSTALL_HEADERS()
