############################################################################
# CMakeLists.txt file for building ROOT misc/minicern package
############################################################################

ROOT_LINKER_LIBRARY(minicern *.c *.f )

# Disable optimization since it some cases was causing crashes
set_target_properties(minicern PROPERTIES COMPILE_FLAGS -O0)
