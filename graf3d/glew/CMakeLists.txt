############################################################################
# CMakeLists.txt file for building ROOT graf3d/glew package
############################################################################

# set variables to emulate find_package(GLEW)
set(GLEW_FOUND TRUE CACHE INTERNAL "" FORCE)
set(GLEW_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/inc CACHE INTERNAL "" FORCE)
set(GLEW_LIBRARY GLEW CACHE INTERNAL "" FORCE)
set(GLEW_LIBRARIES GLEW CACHE INTERNAL "" FORCE)

ROOT_LINKER_LIBRARY(GLEW
  src/glew.c
  LIBRARIES
    OpenGL::GL
    OpenGL::GLU
)

target_include_directories(GLEW PRIVATE ${GLEW_INCLUDE_DIR})

# Do not install headers for all platforms and configurations.
if(UNIX)
  set(installoptions FILTER "wglew")
endif()
if(NOT x11)
  set(installoptions ${installoptions} FILTER "glxew")
endif()

ROOT_INSTALL_HEADERS(${installoptions})
