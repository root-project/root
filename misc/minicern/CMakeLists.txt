############################################################################
# CMakeLists.txt file for building ROOT misc/minicern package
############################################################################

set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fPIC")
ROOT_LINKER_LIBRARY(minicern *.c *.f TYPE STATIC NOINSTALL)
target_link_libraries(minicern ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})

# Disable optimization since it some cases was causing crashes.
# Disable warnings, since what has worked for 40 years...
# (see https://sft.its.cern.ch/jira/browse/ROOT-9179 for the warnings)
set_target_properties(minicern PROPERTIES COMPILE_FLAGS "-O0 -w")



