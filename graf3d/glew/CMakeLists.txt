############################################################################
# CMakeLists.txt file for building ROOT graf3d/glew package
############################################################################

include_directories(${OPENGL_INCLUDE_DIR})

set(CMAKE_C_FLAGS "${GLEW_C_FLAGS} -isystem=${CMAKE_CURRENT_SOURCE_DIR}/isystem")
ROOT_LINKER_LIBRARY(GLEW *.c LIBRARIES ${OPENGL_LIBRARIES})
ROOT_INSTALL_HEADERS()
