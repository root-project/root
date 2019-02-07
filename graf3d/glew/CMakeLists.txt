############################################################################
# CMakeLists.txt file for building ROOT graf3d/glew package
############################################################################

ROOT_LINKER_LIBRARY(GLEW
  src/glew.c
  LIBRARIES
    OpenGL::GL
    OpenGL::GLU
)

target_include_directories(GLEW PRIVATE inc)

# Do not install headers for all platforms and configurations.
if(UNIX)
  set(installoptions FILTER "wglew")
endif()
if(NOT x11)
  set(installoptions ${installoptions} FILTER "glxew")
endif()

ROOT_INSTALL_HEADERS(${installoptions})
