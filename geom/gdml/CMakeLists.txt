############################################################################
# CMakeLists.txt file for building ROOT geom/gdml package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Gdml
  HEADERS
    TGDMLParse.h
    TGDMLWrite.h
  SOURCES
    src/TGDMLParse.cxx
    src/TGDMLWrite.cxx
  DEPENDENCIES
    Geom
    XMLIO
    Hist
    RIO
)
