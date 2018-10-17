############################################################################
# CMakeLists.txt file for building ROOT gui/canvaspainter package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTWebDisplay
    HEADERS 
       ROOT/RWebDisplayHandle.hxx
       ROOT/RWebWindow.hxx
       ROOT/RWebWindowsManager.hxx
    SOURCES
       RWebDisplayHandle.cxx
       RWebWindow.cxx
       RWebWindowsManager.cxx
    LIBRARIES 
       Core 
       RHTTP 
       MathCore
)
