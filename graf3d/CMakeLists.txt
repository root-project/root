if(cxxmodules AND APPLE)
  # FIXME: Any header of graf3d can trigger building of the OSX OpenGL module.
  # It looks like our glew infrastructure combined with the OpenGL modulemap
  # cannot be compiled with -fmodules-local-submodule-visibility yet.
  # Strip out that flag.
  string(REPLACE "-Xclang -fmodules-local-submodule-visibility" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

add_subdirectory(g3d) # special CMakeLists.txt
if(NOT WIN32 AND x11)
  add_subdirectory(x3d) # special CMakeLists.txt
endif()

add_subdirectory(csg) # special CMakeLists.txt

if (opengl)
   add_subdirectory(eve) # special CMakeLists.txt
   add_subdirectory(gl) # special CMakeLists.txt
   if(builtin_glew)
      add_subdirectory(glew)
   endif()
   if(builtin_ftgl)
      add_subdirectory(ftgl)
   endif()
  add_subdirectory(gviz3d) # special CMakeLists.txt
endif()

if(webgui AND root7)
  add_subdirectory(eve7) # special CMakeLists.txt
endif()
