add_subdirectory (geom)        # special CMakeLists.txt
add_subdirectory (geombuilder) # special CMakeLists.txt
add_subdirectory (geompainter) # special CMakeLists.txt
if(gdml)
  add_subdirectory(gdml)
endif()

