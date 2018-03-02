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

ROOT_STANDARD_LIBRARY_PACKAGE(RInterface
                              HEADERS ${R_HEADERS}
                              LIBRARIES ${R_LIBRARIES} readline
                              DEPENDENCIES Core Matrix Thread RIO)
