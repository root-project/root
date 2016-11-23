Add_Subdirectory(hist)             # special CMakeLists.txt
Add_Subdirectory(histpainter)      # special CMakeLists.txt
Add_Subdirectory(spectrum)
Add_Subdirectory(spectrumpainter)  # special CMakeLists.txt
Add_Subdirectory(unfold)
If(CMAKE_Fortran_COMPILER)
  Add_Subdirectory(hbook)
EndIf(CMAKE_Fortran_COMPILER)

