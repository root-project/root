############################################################################
# CMakeLists.txt file for building ROOT rootx package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(core)

ROOT_EXECUTABLE(root *.cxx  ${CMAKE_SOURCE_DIR}/core/clib/src/strlcpy.c 
                LIBRARIES ${X11_LIBRARIES} ${X11_Xpm_LIB} ${X11_Xft_LIB} ${X11_Xext_LIB})

