ROOT_USE_PACKAGE(graf2d/gpad)
ROOT_USE_PACKAGE(graf2d/graf)

if(builtin_freetype)
  add_subdirectory(freetype)
endif()
add_subdirectory(gpad)
add_subdirectory(graf)
add_subdirectory(postscript)
add_subdirectory(mathtext)
if(WIN32)
  add_subdirectory(win32gdk)
else()
  add_subdirectory(x11)         # special CMakeLists.txt
  add_subdirectory(x11ttf)      # special CMakeLists.txt
endif()
add_subdirectory(asimage)     # special CMakeLists.txt
if(qt)
  add_subdirectory(qt)
endif()
if(gviz)
  add_subdirectory(gviz)
endif()
if(fitsio)
  add_subdirectory(fitsio)
endif()
if(cocoa)
  add_subdirectory(quartz)
  add_subdirectory(cocoa)
endif()


