############################################################################
# CMakeLists.txt file for building ROOT MPI package
############################################################################
#Autor: Omar Andres Zapata Mesa 2016
# Omar.Zapata@cern.ch
# http://oproject.org

set(libname RMpi)
include_directories(inc/)

if(python OR python2 OR python3)
    if(PYTHONLIBS_FOUND)
        include_directories(${PYTHON_INCLUDE_DIRS})
        ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -DPYTHON_FOUND)
        add_definitions("-DPYTHON_FOUND" )
    endif()
endif()


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
         Mpi/TPort.h
         Mpi/TMpiTimer.h
         Mpi/TMpiFile.h
         Mpi/TErrorHandler.h
		 TMpi.h
		 )

ROOT_GENERATE_DICTIONARY(G__RMpi ${RMPI_HEADERS} MODULE ${libname} LINKDEF LinkDef.h)

if(python OR python2 OR python3)
    ROOT_LINKER_LIBRARY(RMpi *.cxx G__RMpi.cxx LIBRARIES ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${PYTHON_LIBRARIES}  DEPENDENCIES Core Thread RIO Net Hist Tree)
else()
    ROOT_LINKER_LIBRARY(RMpi *.cxx G__RMpi.cxx LIBRARIES ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} DEPENDENCIES Core Thread RIO Net Hist Tree)
endif()
ROOT_INSTALL_HEADERS()






