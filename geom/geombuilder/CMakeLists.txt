############################################################################
# CMakeLists.txt file for building ROOT geom/geom package
# @author Pere Mato, CERN
############################################################################

set(libname GeomBuilder)

ROOT_USE_PACKAGE(geom/geom)
ROOT_USE_PACKAGE(gui/gui)
ROOT_USE_PACKAGE(gui/ged)

set(headers TGeoVolumeEditor.h TGeoBBoxEditor.h TGeoMediumEditor.h
            TGeoNodeEditor.h TGeoMatrixEditor.h TGeoManagerEditor.h
            TGeoTubeEditor.h TGeoConeEditor.h TGeoTrd1Editor.h
            TGeoTrd2Editor.h TGeoMaterialEditor.h TGeoTabManager.h
            TGeoSphereEditor.h TGeoPconEditor.h TGeoParaEditor.h
            TGeoTorusEditor.h TGeoEltuEditor.h TGeoHypeEditor.h
            TGeoPgonEditor.h TGeoTrapEditor.h TGeoGedFrame.h )

ROOT_GENERATE_DICTIONARY(G__${libname} ${headers} LINKDEF LinkDef.h)

ROOT_GENERATE_ROOTMAP(${libname} LINKDEF LinkDef.h DEPENDENCIES Geom Graf3d Gpad Graf Gui Ged)
ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx LIBRARIES Geom Graf3d Gpad Graf Gui Ged)
ROOT_INSTALL_HEADERS()

