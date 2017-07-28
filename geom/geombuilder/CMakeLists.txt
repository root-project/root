############################################################################
# CMakeLists.txt file for building ROOT geom/geom package
# @author Pere Mato, CERN
############################################################################

set(headers TGeoVolumeEditor.h TGeoBBoxEditor.h TGeoMediumEditor.h
            TGeoNodeEditor.h TGeoMatrixEditor.h TGeoManagerEditor.h
            TGeoTubeEditor.h TGeoConeEditor.h TGeoTrd1Editor.h
            TGeoTrd2Editor.h TGeoMaterialEditor.h TGeoTabManager.h
            TGeoSphereEditor.h TGeoPconEditor.h TGeoParaEditor.h
            TGeoTorusEditor.h TGeoEltuEditor.h TGeoHypeEditor.h
            TGeoPgonEditor.h TGeoTrapEditor.h TGeoGedFrame.h )

ROOT_STANDARD_LIBRARY_PACKAGE(GeomBuilder
                              HEADERS ${headers}
                              DEPENDENCIES Geom Graf3d Gpad Graf Gui Ged)
