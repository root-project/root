############################################################################
# CMakeLists.txt file for building ROOT R package
############################################################################
#Autor: Omar Andres Zapata Mesa 31/05/2013,14/07/2014

set(libname RInterface)

include_directories(SYSTEM ${R_INCLUDE_DIRS})

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-overloaded-virtual)

set(R_HEADERS ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRInterface.h
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRObject.h 
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRFunctionExport.h
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRFunctionImport.h
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRInternalFunction.h
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRDataFrame.h
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/RExports.h)

ROOT_GENERATE_DICTIONARY(G__RInterface ${R_HEADERS} MODULE ${libname} LINKDEF LinkDef.h DEPENDENCIES Core Matrix Thread RIO)
ROOT_LINKER_LIBRARY(RInterface *.cxx G__RInterface.cxx LIBRARIES ${R_LIBRARIES} DEPENDENCIES Core Matrix Thread RIO readline)

ROOT_INSTALL_HEADERS()
