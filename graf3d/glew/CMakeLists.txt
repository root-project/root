############################################################################
# CMakeLists.txt file for building ROOT graf3d/glew package
############################################################################

ROOT_LINKER_LIBRARY(GLEW
  src/glew.c
  LIBRARIES
    ${OPENGL_LIBRARIES}
)

target_include_directories(GLEW PUBLIC ${OPENGL_INCLUDE_DIR} PRIVATE inc)

# Do not install headers for all platforms and configurations.
if(UNIX)
  set(installoptions FILTER "wglew")
endif()
if(NOT x11)
  set(installoptions ${installoptions} FILTER "glxew")
endif()

ROOT_INSTALL_HEADERS(${installoptions})
