add_subdirectory (geom)        # special CMakeLists.txt
add_subdirectory (geombuilder) # special CMakeLists.txt
add_subdirectory (geompainter) # special CMakeLists.txt
if(gdml)
  add_subdirectory(gdml)
  if(NOT gnuinstall)
    install(DIRECTORY gdml/ DESTINATION geom/gdml
            FILES_MATCHING PATTERN "*.py"
            PATTERN "inc" EXCLUDE
            PATTERN "src" EXCLUDE)
  endif()
endif()
if(geocad)
  add_subdirectory(geocad)
endif()

