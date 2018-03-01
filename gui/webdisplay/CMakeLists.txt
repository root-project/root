############################################################################
# CMakeLists.txt file for building ROOT gui/canvaspainter package
############################################################################

set(libname ROOTWebDisplay)

ROOT_GLOB_HEADERS(WebDisplay_headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT/*.hxx)
ROOT_GLOB_SOURCES(WebDisplay_sources src/*.cxx)

ROOT_GENERATE_DICTIONARY(G__${libname} ${WebDisplay_headers} MODULE ${libname} LINKDEF ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef.h OPTIONS DEPENDENCIES RHTTP)

ROOT_OBJECT_LIBRARY(ROOTWebDisplayObjs G__ROOTWebDisplay.cxx ${WebDisplay_sources})

ROOT_LINKER_LIBRARY(${libname} $<TARGET_OBJECTS:ROOTWebDisplayObjs> $<TARGET_OBJECTS:RootPcmObjs>
                    LIBRARIES ${CMAKE_DL_LIBS}
                    DEPENDENCIES RHTTP MathCore)

ROOT_INSTALL_HEADERS()
