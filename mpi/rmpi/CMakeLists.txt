############################################################################
# CMakeLists.txt file for building ROOT MPI package
############################################################################
#Autor: Omar Andres Zapata Mesa 2016
# Omar.Zapata@cern.ch
# http://oproject.org

set(libname RMpi)

set(RMPI_HEADERS ${CMAKE_SOURCE_DIR}/mpi/rmpi/inc/Mpi/TEnvironment.h
		 ${CMAKE_SOURCE_DIR}/mpi/rmpi/inc/Mpi/TMpiMessage.h
		 ${CMAKE_SOURCE_DIR}/mpi/rmpi/inc/Mpi/TCommunicator.h
		 ${CMAKE_SOURCE_DIR}/mpi/rmpi/inc/Mpi/Types.h
		 ${CMAKE_SOURCE_DIR}/mpi/rmpi/inc/Mpi.h
		 )

ROOT_GENERATE_DICTIONARY(G__RMpi ${RMPI_HEADERS} MODULE ${libname} LINKDEF LinkDef.h)
ROOT_LINKER_LIBRARY(RMpi *.cxx G__RMpi.cxx LIBRARIES ${MPI_CXX_LIBRARIES} DEPENDENCIES Core Thread RIO Net)

ROOT_INSTALL_HEADERS()






