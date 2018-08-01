add_root_subdirectory(hist)             # special CMakeLists.txt
add_root_subdirectory(histdraw)         # special CMakeLists.txt
add_root_subdirectory(histpainter)      # special CMakeLists.txt
add_root_subdirectory(spectrum)
add_root_subdirectory(spectrumpainter)  # special CMakeLists.txt
if(xml)
  add_root_subdirectory(unfold)
endif()
if(CMAKE_Fortran_COMPILER)
  add_root_subdirectory(hbook)
endif()

