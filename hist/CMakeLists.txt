add_subdirectory(hist)             # special CMakeLists.txt
add_subdirectory(histdraw)         # special CMakeLists.txt
add_subdirectory(histpainter)      # special CMakeLists.txt
if (spectrum)
   add_subdirectory(spectrum)
   add_subdirectory(spectrumpainter)  # special CMakeLists.txt
endif()
if(xml)
  add_subdirectory(unfold)
endif()
if(CMAKE_Fortran_COMPILER)
  add_subdirectory(hbook)
endif()

