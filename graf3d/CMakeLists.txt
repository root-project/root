add_root_subdirectory(g3d) # special CMakeLists.txt
if(NOT WIN32 AND x11)
  add_root_subdirectory(x3d) # special CMakeLists.txt
endif()
if (opengl)
   add_root_subdirectory(eve) # special CMakeLists.txt
   add_root_subdirectory(gl) # special CMakeLists.txt
   if(builtin_glew)
      add_root_subdirectory(glew)
   endif()
   if(builtin_ftgl)
      add_root_subdirectory(ftgl)
   endif()
  add_root_subdirectory(gviz3d) # special CMakeLists.txt
endif()

