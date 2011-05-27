############################################################################
# CMakeLists.txt file for building ROOT graf3d/glew package
############################################################################

include_directories(${OPENGL_INCLUDE_DIR})

ROOT_LINKER_LIBRARY(GLEW *.c LIBRARIES ${OPENGL_LIBRARIES})
ROOT_INSTALL_HEADERS()
