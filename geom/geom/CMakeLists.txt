############################################################################
# CMakeLists.txt file for building ROOT geom/geom package
############################################################################

set(headers1 TGeoAtt.h TGeoStateInfo.h TGeoBoolNode.h
             TGeoMedium.h TGeoMaterial.h
             TGeoMatrix.h TGeoVolume.h TGeoNode.h
             TGeoVoxelFinder.h TGeoShape.h TGeoBBox.h
             TGeoPara.h TGeoTube.h TGeoTorus.h TGeoSphere.h
             TGeoEltu.h TGeoHype.h TGeoCone.h TGeoPcon.h
             TGeoPgon.h TGeoArb8.h TGeoTrd1.h TGeoTrd2.h
             TGeoManager.h TGeoCompositeShape.h TGeoShapeAssembly.h
             TGeoScaledShape.h TVirtualGeoPainter.h TVirtualGeoTrack.h
             TVirtualGeoConverter.h TGeoPolygon.h TGeoXtru.h TGeoPhysicalNode.h
             TGeoHelix.h TGeoParaboloid.h TGeoElement.h TGeoHalfSpace.h
             TGeoBuilder.h TGeoNavigator.h)
set(headers2 TGeoPatternFinder.h TGeoCache.h TVirtualMagField.h
             TGeoUniformMagField.h TGeoGlobalMagField.h TGeoBranchArray.h
             TGeoExtension.h TGeoParallelWorld.h)


ROOT_STANDARD_LIBRARY_PACKAGE(Geom
                              HEADERS ${headers1} ${headers2}
                              DEPENDENCIES Thread RIO MathCore)
