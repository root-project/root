############################################################################
# CMakeLists.txt file for building ROOT graf3d/ftgl package
############################################################################

include_directories(${OPENGL_INCLUDE_DIR} ${FREETYPE_INCLUDE_DIRS})
if(MACOSX_GLU_DEPRECATED)
  add_definitions(-Wno-deprecated-declarations)
endif()

ROOT_LINKER_LIBRARY(FTGL *.cxx LIBRARIES ${OPENGL_LIBRARIES} ${FREETYPE_LIBRARIES} ${ZLIB_LIBRARY})
if(builtin_freetype)
  add_dependencies(FTGL FREETYPE)
endif()
ROOT_INSTALL_HEADERS()

