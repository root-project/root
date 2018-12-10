############################################################################
# CMakeLists.txt file for building ROOT graf2d/cocoa package
# @author Pere Mato, CERN
############################################################################

add_definitions("-ObjC++")

include_directories(${FREETYPE_INCLUDE_DIRS})

ROOT_STANDARD_LIBRARY_PACKAGE(GCocoa
                              HEADERS T*.h
                              SOURCES *.mm
                              DEPENDENCIES Gui GQuartz
                              LIBRARIES  "-framework Cocoa" "-framework OpenGL" ${FREETYPE_LIBRARIES})
if(NOT opengl) # special case when cocoa is enabled but not opengl (i.e. gminimal=ON)
  set_source_files_properties(src/TGCocoa.mm PROPERTIES 
                             COMPILE_FLAGS "-I${CMAKE_SOURCE_DIR}/graf3d/gl/inc -I${CMAKE_SOURCE_DIR}/graf3d/glew/inc")
endif()
# We know GL is deprecated on macOS. And our answer is WebGUI/Graphics.
set_target_properties(GCocoa PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")
