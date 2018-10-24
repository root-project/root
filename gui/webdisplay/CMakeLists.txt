############################################################################
# CMakeLists.txt file for building ROOT gui/webdisplay package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTWebDisplay
    HEADERS 
       ROOT/RWebDisplayArgs.hxx
       ROOT/RWebDisplayHandle.hxx
       ROOT/RWebWindow.hxx
       ROOT/RWebWindowsManager.hxx
    SOURCES
       RWebDisplayArgs.cxx
       RWebDisplayHandle.cxx
       RWebWindow.cxx
       RWebWindowsManager.cxx
    LIBRARIES 
       Core 
       RHTTP 
       MathCore
)
