############################################################################
# CMakeLists.txt file for building ROOT graf3d/gl package
############################################################################
ROOT_USE_PACKAGE(gui/ged)
if(builtin_glew)
  ROOT_USE_PACKAGE(graf3d/glew)
endif()

include_directories(${OPENGL_INCLUDE_DIR} ${FTGL_INCLUDE_DIR} ${FREETYPE_INCLUDE_DIRS})

ROOT_GLOB_HEADERS(headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
list(REMOVE_ITEM headers  gl2ps.h CsgOps.h TGLIncludes.h TGLWSIncludes.h TGLContextPrivate.h
                     TGLMarchingCubes.h TKDEAdapter.h TGL5DPainter.h TKDEFGT.h TGLIsoMesh.h)
ROOT_GLOB_SOURCES(sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)

if(WIN32 OR cocoa)
  list(REMOVE_ITEM headers TX11GL.h)
  list(REMOVE_ITEM sources TX11GL.cxx)
  set(installoptions OPTIONS REGEX "TX11GL" EXCLUDE)
endif()

if(WIN32)
  set(gllibs opengl32.lib glu32.lib)
elseif(cocoa)
  set(gllibs "-framework OpenGL")
else()
  set(gllibs ${OPENGL_LIBRARIES} ${X11_LIBRARIES})
endif()

if(MACOSX_GLU_DEPRECATED)
  add_definitions(-Wno-deprecated-declarations)
endif()

set_source_files_properties(src/TGLFontManager.cxx PROPERTIES COMPILE_FLAGS "${FTGL_CFLAGS}")
set_source_files_properties(src/TGLText.cxx PROPERTIES COMPILE_FLAGS "${FTGL_CFLAGS}")

ROOT_GENERATE_DICTIONARY(G__GL ${headers} MODULE RGL LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")

ROOT_LINKER_LIBRARY(RGL ${sources} G__GL.cxx  LIBRARIES ${gllibs} GLEW ${FTGL_LIBRARIES} DEPENDENCIES Hist Gui Ged)

ROOT_INSTALL_HEADERS(${installoptions})
