############################################################################
# CMakeLists.txt file for building ROOT graf3d/gl package
############################################################################
ROOT_USE_PACKAGE(gui/ged)
if(builtin_glew)
  ROOT_USE_PACKAGE(graf3d/glew)
endif()
if(builtin_ftgl)
  ROOT_USE_PACKAGE(graf3d/ftgl)
endif()
include_directories(${OPENGL_INCLUDE_DIR})

ROOT_GLOB_HEADERS(headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
list(REMOVE_ITEM headers  gl2ps.h CsgOps.h TGLIncludes.h TGLWSIncludes.h TGLContextPrivate.h 
                     TGLMarchingCubes.h TKDEAdapter.h TGL5DPainter.h TKDEFGT.h TGLIsoMesh.h)
ROOT_GLOB_SOURCES(sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)

if(WIN32)
  list(REMOVE_ITEM headers TX11GL.h)
  list(REMOVE_ITEM sources TX11GL.cxx)  
endif()

set_source_files_properties(src/TGLFontManager.cxx PROPERTIES COMPILE_FLAGS "-I${FREETYPE_INCLUDE_DIR} -DBUILTIN_FTGL")
set_source_files_properties(src/TGLText.cxx PROPERTIES COMPILE_FLAGS "-I${FREETYPE_INCLUDE_DIR} -DBUILTIN_FTGL")

ROOT_GENERATE_DICTIONARY(G__GL ${headers} LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(RGL LINKDEF LinkDef.h DEPENDENCIES Graf3d Gui Graf Hist Ged MathCore Tree TreePlayer)
ROOT_LINKER_LIBRARY(RGL ${sources} G__GL.cxx  LIBRARIES ${OPENGL_LIBRARIES} GLEW FTGL DEPENDENCIES Hist Gui Ged)

ROOT_INSTALL_HEADERS()
