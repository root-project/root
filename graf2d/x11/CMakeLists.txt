############################################################################
# CMakeLists.txt file for building ROOT graf2d/x11 package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(GX11
                              HEADERS T*.h
                              SOURCES *.cxx  *.c
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES Core
                              LIBRARIES Core ${X11_LIBRARIES} ${X11_Xpm_LIB} ${CMAKE_THREAD_LIBS_INIT}
                              INSTALL_OPTIONS FILTER "inc/X11/" FILTER "X11$")

if(AIX)
  target_include_directories(GX11 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
endif()
