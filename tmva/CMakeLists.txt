add_subdirectory(tmva)
add_subdirectory(tmvagui)

# Build pyMVA. Allows users to call python mva's from c++ ROOT/TMVA.
if(python AND tmva-pymva AND NUMPY_FOUND)
   add_subdirectory(pymva)
endif()
if(r)
add_subdirectory(rmva)
endif()
