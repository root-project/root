############################################################################
# CMakeLists.txt file for building ROOT graf3d/glew package
############################################################################

include_directories(${OPENGL_INCLUDE_DIR})

# Do not install headers for all platforms and configurations.
if(UNIX)
  set(installoptions OPTIONS REGEX "wglew" EXCLUDE)
else() # Windows
  set(installoptions OPTIONS REGEX "glew" EXCLUDE)
endif()
if(NOT x11)
  set(installoptions ${installoptions} OPTIONS REGEX "glxew" EXCLUDE)
endif()

ROOT_LINKER_LIBRARY(GLEW *.c LIBRARIES ${OPENGL_LIBRARIES})
ROOT_INSTALL_HEADERS(${installoptions})
