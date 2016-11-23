############################################################################
# CMakeLists.txt file for building ROOT R package
############################################################################
#Autor: Omar Andres Zapata Mesa 31/05/2013,14/07/2014

# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/r")

set(libname RMVA)

include_directories(${R_INCLUDE_DIRS})

set(R_HEADERS ${CMAKE_SOURCE_DIR}/tmva/rmva/inc/TMVA/RMethodBase.h 
              ${CMAKE_SOURCE_DIR}/tmva/rmva/inc/TMVA/MethodC50.h
              ${CMAKE_SOURCE_DIR}/tmva/rmva/inc/TMVA/MethodRSNNS.h
              ${CMAKE_SOURCE_DIR}/tmva/rmva/inc/TMVA/MethodRSVM.h
              ${CMAKE_SOURCE_DIR}/tmva/rmva/inc/TMVA/MethodRXGB.h
              )

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-overloaded-virtual)
ROOT_GENERATE_DICTIONARY(G__RMVA ${R_HEADERS} MODULE ${libname} LINKDEF LinkDef.h)
ROOT_LINKER_LIBRARY(RMVA *.cxx G__RMVA.cxx LIBRARIES ${R_LIBRARIES}  DEPENDENCIES Core Matrix Thread RIO RInterface TMVA)

ROOT_INSTALL_HEADERS()





