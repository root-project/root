############################################################################
# CMakeLists.txt file for building ROOT graf3d/glew package
############################################################################

include_directories(${OPENGL_INCLUDE_DIR})

# Do not install headers for all platforms and configurations.
if(UNIX)
  set(installoptions FILTER "wglew")
endif()
if(NOT x11)
  set(installoptions ${installoptions} FILTER "glxew")
endif()

ROOT_LINKER_LIBRARY(GLEW *.c LIBRARIES ${OPENGL_LIBRARIES})
ROOT_INSTALL_HEADERS(${installoptions})
