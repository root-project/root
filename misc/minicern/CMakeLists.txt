############################################################################
# CMakeLists.txt file for building ROOT misc/minicern package
############################################################################

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fPIC")
ROOT_LINKER_LIBRARY(minicern *.c *.f TYPE STATIC)

# Disable optimization since it some cases was causing crashes
set_target_properties(minicern PROPERTIES COMPILE_FLAGS -O0)


