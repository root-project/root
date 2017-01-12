############################################################################
# CMakeLists.txt file for building ROOTMpi tests.
# @author Omar Zapata
############################################################################

project(mpi-tests)
find_package(ROOT REQUIRED)
find_package(MPI REQUIRED)

#ROOTMpi binary to execute processes
set(ROOTMPIEXEC ${ROOT_BINARY_DIR}/rootmpi)

function(ROOT_ADD_TEST_MPI name nprocs)
  set(test_parameters -np ${nprocs} "${CMAKE_CURRENT_SOURCE_DIR}/${name}.C")
  ROOT_ADD_TEST("test-rootmpi-${name}" COMMAND ${ROOTMPIEXEC} ${test_parameters})
endfunction(ROOT_ADD_TEST_MPI)

ROOT_ADD_TEST_MPI(p2p 2)
ROOT_ADD_TEST_MPI(p2p_nonblocking 2)
ROOT_ADD_TEST_MPI(ibcast 4)
