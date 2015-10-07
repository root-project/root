############################################################################
# CMakeLists.txt file for building ROOT R package
############################################################################
#Autor: Omar Andres Zapata Mesa 31/05/2013,14/07/2014

# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/r")

set(libname RInterface)

include_directories(${R_INCLUDE_DIRS})

set(R_HEADERS ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRInterface.h
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRObjectProxy.h 
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRFunction.h
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRInternalFunction.h
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRDataFrame.h
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/RExports.h)
set(RHIST_HEADERS ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRF1.h              
                  ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRGraph.h)
set(ROI_HEADERS ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRFile.h)
set(RGPAD_HEADERS ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRCanvas.h)
set(RRINT_HEADERS ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRRint.h)

ROOT_GENERATE_DICTIONARY(G__RInterface ${R_HEADERS} MODULE ${libname} LINKDEF LinkDef.h)
ROOT_LINKER_LIBRARY(RInterface *.cxx G__RInterface.cxx LIBRARIES ${R_LIBRARIES} DEPENDENCIES Core Matrix Thread RIO readline)

ROOT_INSTALL_HEADERS()






