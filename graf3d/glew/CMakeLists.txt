############################################################################
# CMakeLists.txt file for building ROOT graf3d/glew package
############################################################################

include_directories(${OPENGL_INCLUDE_DIR})

# We need to disallow the direct use of gl.h. This way people will see the error
# and the suggested fix. This happens by providing our own "fake" system gl.h.
set(CMAKE_C_FLAGS " -isystem ${CMAKE_CURRENT_SOURCE_DIR}/isystem ${GLEW_C_FLAGS}")
ROOT_LINKER_LIBRARY(GLEW *.c LIBRARIES ${OPENGL_LIBRARIES})
ROOT_INSTALL_HEADERS()
