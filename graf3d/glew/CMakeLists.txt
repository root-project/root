############################################################################
# CMakeLists.txt file for building ROOT graf3d/glew package
############################################################################

include_directories(${OPENGL_INCLUDE_DIR})


ROOT_LINKER_LIBRARY(GLEW *.c LIBRARIES ${OPENGL_LIBRARIES})
ROOT_INSTALL_HEADERS()

if(cxxmodules)
  # For some reason cmake decides to escape the spaces contained in CMAKE_CXX_FLAGS,
  # which makes the compiler invocation invalid. Prevent that by making the list
  # comma separated, which gets translated by cmake into spaces later on.
  string(REPLACE " " ";" GLEW_C_FLAGS ${CMAKE_C_FLAGS})
  add_custom_target(glewcmodule ALL
    DEPENDS move_headers
    COMMAND ${CMAKE_CXX_COMPILER} ${GLEW_C_FLAGS} -I${ROOT_INCLUDE_DIR} -fno-implicit-module-maps -xc -c ${ROOT_INCLUDE_DIR}/TGLIncludes.h)
endif(cxxmodules)
