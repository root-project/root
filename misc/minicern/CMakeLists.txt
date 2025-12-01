# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT misc/minicern package
############################################################################

ROOT_LINKER_LIBRARY(minicern *.c *.f TYPE STATIC)
set_property(TARGET minicern PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(minicern ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})

# set_target_properties(minicern PROPERTIES COMPILE_FLAGS "-fsanitize=undefined -fsanitize=address") # TODO enabling this shows issues
# target_link_options(minicern BEFORE PUBLIC -fsanitize=undefined PUBLIC -fsanitize=address) # TODO enabling this shows issues
