#---FIXME: Disable modules build for graf3d until the glew.h issue gets fixed---
if(cxxmodules)
  if(ROOT_ARCHITECTURE MATCHES macosx)
    string(REPLACE "${ROOT_CXXMODULES_FLAGS}" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "${ROOT_CXXMODULES_FLAGS}" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
  endif()
endif(cxxmodules)

add_subdirectory(g3d) # special CMakeLists.txt
if(NOT WIN32 AND x11)
  add_subdirectory(x3d) # special CMakeLists.txt
endif()
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

