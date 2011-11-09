############################################################################
# CMakeLists.txt file for building ROOT graf3d/ftgl package
############################################################################

include_directories(${OPENGL_INCLUDE_DIR} ${FREETYPE_INCLUDE_DIRS})

ROOT_LINKER_LIBRARY(FTGL *.cxx LIBRARIES ${OPENGL_LIBRARIES} ${FREETYPE_LIBRARIES})
if(builtin_freetype)
  add_dependencies(FTGL FREETYPE)
endif()
ROOT_INSTALL_HEADERS()

