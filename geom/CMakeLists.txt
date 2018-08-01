add_root_subdirectory (geom)        # special CMakeLists.txt
add_root_subdirectory (geombuilder) # special CMakeLists.txt
add_root_subdirectory (geompainter) # special CMakeLists.txt
if(gdml)
  add_root_subdirectory(gdml)
  if(NOT gnuinstall)
    install(DIRECTORY gdml/ DESTINATION geom/gdml
            FILES_MATCHING PATTERN "*.py"
            PATTERN "inc" EXCLUDE
            PATTERN "src" EXCLUDE)
  endif()
endif()
if(geocad)
  add_root_subdirectory(geocad)
endif()
if(vecgeom)
  add_root_subdirectory(vecgeom)
endif()
  

