############################################################################
# CMakeLists.txt file for building ROOT graf3d/gviz3d package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Gviz3d
  HEADERS
    TStructNode.h
    TStructNodeEditor.h
    TStructNodeProperty.h
    TStructViewer.h
    TStructViewerGUI.h
  SOURCES
    src/TStructNode.cxx
    src/TStructNodeEditor.cxx
    src/TStructNodeProperty.cxx
    src/TStructViewer.cxx
    src/TStructViewerGUI.cxx
  LIBRARIES
    GLEW ${FTGL_LIBRARIES}
  DEPENDENCIES
    Gui
    Ged
    Geom
    RGL
)
