############################################################################
# CMakeLists.txt file for building ROOTMpi tests.
# @author Omar Zapata
############################################################################

project(mpi-tests)
find_package(ROOT REQUIRED)
find_package(MPI REQUIRED)

#ROOTMpi binary to execute processes
#Standalone Build
if(CMAKE_PROJECT_NAME STREQUAL ROOTMpi)
    set(ROOTMPIEXEC ${CMAKE_BINARY_DIR}/rootmpi/rootmpi)
else()
    set(ROOTMPIEXEC ${ROOT_BINARY_DIR}/rootmpi)
endif()

function(ROOT_ADD_TEST_MPI name nprocs)
  set(test_parameters -np ${nprocs} -b "${CMAKE_CURRENT_SOURCE_DIR}/${name}.C")
  ROOT_ADD_TEST("test-rootmpi-${name}" COMMAND ${ROOTMPIEXEC} ${test_parameters})
endfunction(ROOT_ADD_TEST_MPI)

function(ROOT_ADD_TEST_MPI_ARG name nprocs arg)
  set(test_parameters -np ${nprocs} -b "${CMAKE_CURRENT_SOURCE_DIR}/${name}.C(${arg})")
  ROOT_ADD_TEST("test-rootmpi-${name}" COMMAND ${ROOTMPIEXEC} ${test_parameters})
endfunction(ROOT_ADD_TEST_MPI_ARG)


ROOT_ADD_TEST_MPI(env 2)
ROOT_ADD_TEST_MPI(info 2)
ROOT_ADD_TEST_MPI(port 2)
ROOT_ADD_TEST_MPI(serialization 2)
ROOT_ADD_TEST_MPI(p2p 2)
ROOT_ADD_TEST_MPI(p2p_nonblocking 2)
ROOT_ADD_TEST_MPI(bcast 4)
ROOT_ADD_TEST_MPI(ibcast 4)
ROOT_ADD_TEST_MPI(scatter 4)
ROOT_ADD_TEST_MPI(gather 4)
ROOT_ADD_TEST_MPI(reduce 4)
ROOT_ADD_TEST_MPI(file 4)
ROOT_ADD_TEST_MPI(communicators 4)
ROOT_ADD_TEST_MPI_ARG(commspawn 4 "\\\"${CMAKE_CURRENT_SOURCE_DIR}/commspawnproc.C\\\"")

if(python OR python2 OR python3)
    if(PYTHONLIBS_FOUND)
        function(ROOT_ADD_TEST_MPI_PY name nprocs)
        set(test_parameters -np ${nprocs} -b "${CMAKE_CURRENT_SOURCE_DIR}/${name}.py")
        ROOT_ADD_TEST("test-rootmpi-${name}-py" COMMAND ${ROOTMPIEXEC} ${test_parameters})
        endfunction(ROOT_ADD_TEST_MPI_PY)
    
    ROOT_ADD_TEST_MPI_PY(p2p 2)

    endif(PYTHONLIBS_FOUND)    
endif()
