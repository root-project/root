############################################################################
# CMakeLists.txt file for building ROOT graf3d/gviz3d package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Gviz3d
                              LIBRARIES GLEW ${FTGL_LIBRARIES}
                              DEPENDENCIES Gui Ged Geom RGL)
