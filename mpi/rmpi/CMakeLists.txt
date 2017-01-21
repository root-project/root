############################################################################
# CMakeLists.txt file for building ROOT MPI package
############################################################################
#Autor: Omar Andres Zapata Mesa 2016
# Omar.Zapata@cern.ch
# http://oproject.org

set(libname RMpi)

set(RMPI_HEADERS Mpi/TEnvironment.h
		 Mpi/TMpiMessage.h
		 Mpi/TCommunicator.h
		 Mpi/TStatus.h
		 Mpi/TRequest.h
                 Mpi/Globals.h
                 Mpi/TOp.h
                 Mpi/TGroup.h
		 Mpi.h
		 )

ROOT_GENERATE_DICTIONARY(G__RMpi ${RMPI_HEADERS} MODULE ${libname} LINKDEF LinkDef.h)
ROOT_LINKER_LIBRARY(RMpi *.cxx G__RMpi.cxx LIBRARIES ${MPI_CXX_LIBRARIES} DEPENDENCIES Core Thread RIO Net)

ROOT_INSTALL_HEADERS()






