############################################################################
# CMakeLists.txt file for building ROOT rootx package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(core)
if(x11)
  include_directories(${X11_INCLUDE_DIR})
  ROOT_EXECUTABLE(root *.cxx  ${CMAKE_SOURCE_DIR}/core/clib/src/strlcpy.c
                  LIBRARIES ${X11_LIBRARIES} ${X11_Xpm_LIB} ${X11_Xft_LIB} ${X11_Xext_LIB})
elseif(cocoa)
  add_custom_target(root ALL 
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/root.exe ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/root)
  add_dependencies(root root.exe)
  set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/root)
  install(PROGRAMS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/root
          DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT applications)

endif()

