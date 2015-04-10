############################################################################
# CMakeLists.txt file for building ROOT math/rtools package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(math/mathcore)
ROOT_USE_PACKAGE(bindings/r)

FIND_PACKAGE(R REQUIRED)
FIND_PACKAGE(Rcpp REQUIRED)
FIND_PACKAGE(RInside REQUIRED)

if(${R_FOUND} AND ${RCPP_FOUND} AND ${RINSIDE_FOUND})

STRING(REGEX REPLACE "-I" "" RCPP_INCLUDE_DIR ${RCPP_INCLUDE_DIR})
STRING(REGEX REPLACE "-I" "" RINSIDE_INCLUDE_DIR  ${RINSIDE_INCLUDE_DIR})
STRING(REGEX REPLACE "-I" "" R_INCLUDE_DIR  ${R_INCLUDE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/bindings/r/inc ${RCPP_INCLUDE_DIR} ${RINSIDE_INCLUDE_DIR} ${R_INCLUDE_DIR})


ROOT_GENERATE_DICTIONARY(G__Rtools Math/RMinimizer.h MODULE Rtools LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(Rtools *.cxx G__Rtools.cxx LIBRARIES Core ${R_LIBRARIES}  ${RINSIDE_LIBRARIES} DEPENDENCIES MathCore RInterface)

ROOT_INSTALL_HEADERS()

FILE(COPY "${CMAKE_SOURCE_DIR}/etc/plugins/ROOT@@Math@@Minimizer/P090_RMinimizer.C" DESTINATION "${CMAKE_BINARY_DIR}/etc/plugins/ROOT@@Math@@Minimizer/")

#ROOT_ADD_TEST_SUBDIRECTORY(test)

endif()
