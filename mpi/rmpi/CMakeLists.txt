############################################################################
# CMakeLists.txt file for building ROOT MPI package
############################################################################
#Autor: Omar Andres Zapata Mesa 2016
# Omar.Zapata@cern.ch
# http://oproject.org

set(libname RMpi)
include_directories(inc/)

set(RMPI_HEADERS Mpi/TEnvironment.h
		 Mpi/TMpiMessage.h
		 Mpi/TCommunicator.h
		 Mpi/TIntraCommunicator.h
		 Mpi/TInterCommunicator.h
		 Mpi/TStatus.h
		 Mpi/TRequest.h
         Mpi/Globals.h
         Mpi/TOp.h
         Mpi/TGroup.h
         Mpi/TInfo.h
		 Mpi.h
		 )

ROOT_GENERATE_DICTIONARY(G__RMpi ${RMPI_HEADERS} MODULE ${libname} LINKDEF LinkDef.h)
ROOT_LINKER_LIBRARY(RMpi *.cxx G__RMpi.cxx LIBRARIES ${MPI_CXX_LIBRARIES} DEPENDENCIES Core Thread RIO Net)

ROOT_INSTALL_HEADERS()






