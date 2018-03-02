############################################################################
# CMakeLists.txt file for building ROOT graf2d/x11ttf package
# @author Pere Mato, CERN
############################################################################

include_directories(${FREETYPE_INCLUDE_DIRS} ${X11_INCLUDE_DIR})

ROOT_STANDARD_LIBRARY_PACKAGE(GX11TTF
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES GX11 Graf
                              LIBRARIES ${X11_LIBRARIES} ${X11_Xpm_LIB} ${X11_Xft_LIB} ${FREETYPE_LIBRARIES} ${ZLIB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
