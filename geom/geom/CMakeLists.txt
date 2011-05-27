############################################################################
# CMakeLists.txt file for building ROOT geom/geom package
############################################################################

set(libname Geom)

ROOT_USE_PACKAGE(math)
ROOT_USE_PACKAGE(io/io)
ROOT_USE_PACKAGE(hist/hist)

set(headers1 TGeoAtt.h TGeoBoolNode.h
             TGeoMedium.h TGeoMaterial.h
             TGeoMatrix.h TGeoVolume.h TGeoNode.h
             TGeoVoxelFinder.h TGeoShape.h TGeoBBox.h
             TGeoPara.h TGeoTube.h TGeoTorus.h TGeoSphere.h
             TGeoEltu.h TGeoHype.h TGeoCone.h TGeoPcon.h 
             TGeoPgon.h TGeoArb8.h TGeoTrd1.h TGeoTrd2.h
             TGeoManager.h TGeoCompositeShape.h TGeoShapeAssembly.h
             TGeoScaledShape.h TVirtualGeoPainter.h TVirtualGeoTrack.h
             TGeoPolygon.h TGeoXtru.h TGeoPhysicalNode.h
             TGeoHelix.h TGeoParaboloid.h TGeoElement.h TGeoHalfSpace.h
             TGeoBuilder.h TGeoNavigator.h)
set(headers2 TGeoPatternFinder.h TGeoCache.h TVirtualMagField.h
             TGeoUniformMagField.h TGeoGlobalMagField.h TGeoBranchArray.h)


ROOT_GENERATE_DICTIONARY(G__${libname}1 ${headers1} LINKDEF LinkDef1.h)
ROOT_GENERATE_DICTIONARY(G__${libname}2 ${headers2} LINKDEF LinkDef2.h)

ROOT_GENERATE_ROOTMAP(${libname} LINKDEF LinkDef1.h  LinkDef2.h DEPENDENCIES RIO MathCore)
ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}1.cxx G__${libname}2.cxx LIBRARIES RIO MathCore)
ROOT_INSTALL_HEADERS()

