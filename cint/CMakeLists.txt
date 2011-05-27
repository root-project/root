# To generate the dictonaries cint_tmp has to run in $ROOTSYS with the include files in $ROOTSYS/include.
# This can't be done in a separate build directory, but what can be done is
# to simulate the setup by copying some cint directories. It looks like this is enough.
execute_process(COMMAND cmake -E make_directory ${CMAKE_BINARY_DIR}/cint/cint 
                COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/cint/cint/include ${CMAKE_BINARY_DIR}/cint/cint/include 
                COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/cint/cint/lib ${CMAKE_BINARY_DIR}/cint/cint/lib
                COMMAND cmake -E copy_directory ${CMAKE_SOURCE_DIR}/cint/cint/stl ${CMAKE_BINARY_DIR}/cint/cint/stl )

add_subdirectory(cint)
add_subdirectory(ROOT)
if(reflex)
  add_subdirectory(reflex)
endif()
if(cintex)
  add_subdirectory(cintex)
endif()

