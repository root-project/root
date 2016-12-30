############################################################################
# CMakeLists.txt file for building ROOT MPI package
############################################################################
#Autor: Omar Andres Zapata Mesa 2016
# Omar.Zapata@cern.ch
# http://oproject.org

set(libname RMpi)

include_directories(${MPI_CXX_INCLUDE_PATH})

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-overloaded-virtual ${MPI_CXX_COMPILE_FLAGS} -Wno-literal-suffix)


set(RMPI_HEADERS ${CMAKE_SOURCE_DIR}/mpi/rmpi/inc/Mpi/TEnvironment.h
		 ${CMAKE_SOURCE_DIR}/mpi/rmpi/inc/Mpi/TMpiMessage.h
		 ${CMAKE_SOURCE_DIR}/mpi/rmpi/inc/Mpi/TCommunicator.h
		 )

ROOT_GENERATE_DICTIONARY(G__RMpi ${RMPI_HEADERS} MODULE ${libname} LINKDEF LinkDef.h)
ROOT_LINKER_LIBRARY(RMpi *.cxx G__RMpi.cxx LIBRARIES ${MPI_CXX_LIBRARIES} DEPENDENCIES Core Thread RIO Net)

ROOT_INSTALL_HEADERS()






