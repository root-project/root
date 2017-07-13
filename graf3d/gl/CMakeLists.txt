############################################################################
# CMakeLists.txt file for building ROOT graf3d/gl package
############################################################################

include_directories(${OPENGL_INCLUDE_DIR} ${FTGL_INCLUDE_DIR} ${FREETYPE_INCLUDE_DIRS})

ROOT_GLOB_HEADERS(headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
list(REMOVE_ITEM headers  CsgOps.h TGLIncludes.h TGLWSIncludes.h
                     TGLMarchingCubes.h TKDEAdapter.h TGL5DPainter.h TKDEFGT.h TGLIsoMesh.h)
ROOT_GLOB_SOURCES(sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)

if(WIN32 OR cocoa)
  list(REMOVE_ITEM headers TX11GL.h)
  list(REMOVE_ITEM sources TX11GL.cxx)
  set(installoptions FILTER "TX11GL")
endif()

if(builtin_gl2ps)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/gl2ps)
else()
  list(REMOVE_ITEM sources gl2ps.cxx)
  include_directories(${GL2PS_INCLUDE_DIRS})
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

ROOT_GENERATE_DICTIONARY(G__RGL ${headers} MODULE RGL LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM" DEPENDENCIES Hist Gui Ged)

ROOT_LINKER_LIBRARY(RGL ${sources} G__RGL.cxx  LIBRARIES ${gllibs} ${GL2PS_LIBRARIES} GLEW ${FTGL_LIBRARIES} DEPENDENCIES Hist Gui Ged)

ROOT_INSTALL_HEADERS(${installoptions})
