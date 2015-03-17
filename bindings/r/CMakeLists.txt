############################################################################
# CMakeLists.txt file for building ROOT R package
############################################################################
#Autor: Omar Andres Zapata Mesa 31/05/2013,14/07/2014

# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/r")

set(libname RInterface)

FIND_PACKAGE(R REQUIRED)
FIND_PACKAGE(Rcpp REQUIRED)
FIND_PACKAGE(RInside REQUIRED)

ROOT_USE_PACKAGE(core)
ROOT_USE_PACKAGE(core/rint)
ROOT_USE_PACKAGE(core/thread)
ROOT_USE_PACKAGE(math/matrix)
ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(graf2d/gpad)
ROOT_USE_PACKAGE(hist/hist)

if(${R_FOUND} AND ${RCPP_FOUND} AND ${RINSIDE_FOUND})

FILE(WRITE ${CMAKE_SOURCE_DIR}/bindings/r/inc/compilerdata.h
     "#define RINCLUDEPATH \" ${R_INCLUDE_DIRS} ${RCPP_INCLUDE_DIRS} ${RINSIDE_INCLUDE_DIRS} \"\n"
     "#define RLINKEDLIBS  \" ${R_LIBRARIES} ${RCPP_LIBRARIES} ${RINSIDE_LIBRARIES} \"\n")

add_definitions(-DUSE_ROOT_ERROR )

STRING(REGEX REPLACE "-I" "" RCPP_INCLUDE_DIR ${RCPP_INCLUDE_DIR})
STRING(REGEX REPLACE "-I" "" RINSIDE_INCLUDE_DIR  ${RINSIDE_INCLUDE_DIR})
STRING(REGEX REPLACE "-I" "" R_INCLUDE_DIR  ${R_INCLUDE_DIR})


include_directories(${CMAKE_SOURCE_DIR}/bindings/r/inc ${RCPP_INCLUDE_DIR} ${RINSIDE_INCLUDE_DIR} ${R_INCLUDE_DIR})


set(R_HEADERS ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRInterface.h 
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRObjectProxy.h 
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRFunction.h
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRInternalFunction.h
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRInternalFunction__ctors.h
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/RExports.h
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/compilerdata.h
              ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRSystem.h              
              )
set(RHIST_HEADERS ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRF1.h              
                  ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRGraph.h              
)              

set(ROI_HEADERS ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRFile.h
)              

set(RGPAD_HEADERS ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRCanvas.h
)

set(RRINT_HEADERS ${CMAKE_SOURCE_DIR}/bindings/r/inc/TRRint.h )


SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)


ROOT_GENERATE_DICTIONARY(G__RInterface ${R_HEADERS} MODULE ${libname} LINKDEF LinkDef.h)
# ROOT_GENERATE_ROOTMAP(RInterface LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(RInterface *.cxx G__RInterface.cxx LIBRARIES ${R_LIBRARIES}  ${RINSIDE_LIBRARIES} DEPENDENCIES Core Matrix Thread RIO readline)
set_target_properties(RInterface PROPERTIES LINK_FLAGS ${RCPP_LIBRARIES})

ROOT_INSTALL_HEADERS()
endif()





