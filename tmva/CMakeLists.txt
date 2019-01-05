add_subdirectory(tmva)
add_subdirectory(tmvagui)

# Build pyMVA. Allows users to call python mva's from c++ ROOT/TMVA.
if(python AND tmva-pymva AND NUMPY_FOUND)
   add_subdirectory(pymva)
endif()

# Build RMVA. Allows users to call R mva's from c++ ROOT/TMVA.
if(r OR tmva-rmva)
   add_subdirectory(rmva)
endif()
