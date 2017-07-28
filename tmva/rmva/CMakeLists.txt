############################################################################
# CMakeLists.txt file for building ROOT R package
############################################################################
#Autor: Omar Andres Zapata Mesa 31/05/2013,14/07/2014

set(libname RMVA)

include_directories(SYSTEM ${R_INCLUDE_DIRS})

set(R_HEADERS ${CMAKE_SOURCE_DIR}/tmva/rmva/inc/TMVA/RMethodBase.h 
              ${CMAKE_SOURCE_DIR}/tmva/rmva/inc/TMVA/MethodC50.h
              ${CMAKE_SOURCE_DIR}/tmva/rmva/inc/TMVA/MethodRSNNS.h
              ${CMAKE_SOURCE_DIR}/tmva/rmva/inc/TMVA/MethodRSVM.h
              ${CMAKE_SOURCE_DIR}/tmva/rmva/inc/TMVA/MethodRXGB.h
              )

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-overloaded-virtual)

ROOT_STANDARD_LIBRARY_PACKAGE(RMVA
                              HEADERS ${R_HEADERS}
                              LIBRARIES ${R_LIBRARIES}
                              DEPENDENCIES Core Matrix Thread RIO RInterface TMVA)
