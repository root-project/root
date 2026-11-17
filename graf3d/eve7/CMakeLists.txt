# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set (EXTRA_DICT_OPTS)
if (runtime_cxxmodules AND WIN32)
  set (EXTRA_DICT_OPTS NO_CXXMODULE)
endif()

if (builtin_glu)
  set(EXTRA_GL_LIBS tessglu)
else ()
  if (WIN32)
    set(EXTRA_GL_LIBS Glu32.lib)
  else()
    set(EXTRA_GL_LIBS OpenGL::GLU)
  endif()
endif()

set(ROOTEVE_HEADERS
  ROOT/REveBox.hxx
  ROOT/REveBoxSet.hxx
  ROOT/REveCalo.hxx
  ROOT/REveCaloData.hxx
  ROOT/REveCamera.hxx
  ROOT/REveChunkManager.hxx
  ROOT/REveClient.hxx
  ROOT/REveCompound.hxx
  ROOT/REveDataCollection.hxx
  ROOT/REveDataProxyBuilderBase.hxx
  ROOT/REveDataSimpleProxyBuilder.hxx
  ROOT/REveDataSimpleProxyBuilderTemplate.hxx
  ROOT/REveDataTable.hxx
  ROOT/REveDigitSet.hxx
  ROOT/REveElement.hxx
  ROOT/REveEllipsoid.hxx
  ROOT/REveFrameBox.hxx
  ROOT/REveGeoOverlaps.hxx
  ROOT/REveGeoPolyShape.hxx
  ROOT/REveGeoShape.hxx
  ROOT/REveGeoShapeExtract.hxx
  ROOT/REveGeoTopNode.hxx
  ROOT/REveGluTess.hxx
  ROOT/REveJetCone.hxx
  ROOT/REveLine.hxx
  ROOT/REveLogo.hxx
  ROOT/REveManager.hxx
  ROOT/REvePathMark.hxx
  ROOT/REvePointSet.hxx
  ROOT/REvePolygonSetProjected.hxx
  ROOT/REveProjectionAxis.hxx
  ROOT/REveProjectionBases.hxx
  ROOT/REveProjectionManager.hxx
  ROOT/REveProjections.hxx
  ROOT/REveRGBAPalette.hxx
  ROOT/REveRenderData.hxx
  ROOT/REveSMorph.hxx
  ROOT/REveScalableStraightLineSet.hxx
  ROOT/REveScene.hxx
  ROOT/REveSceneInfo.hxx
  ROOT/REveSecondarySelectable.hxx
  ROOT/REveSelection.hxx
  ROOT/REveShape.hxx
  ROOT/REveStraightLineSet.hxx
  ROOT/REveSystem.hxx
  ROOT/REveTableInfo.hxx
  ROOT/REveTableProxyBuilder.hxx
  ROOT/REveText.hxx
  ROOT/REveTrack.hxx
  ROOT/REveTrackProjected.hxx
  ROOT/REveTrackPropagator.hxx
  ROOT/REveTrans.hxx
  ROOT/REveTreeTools.hxx
  ROOT/REveTypes.hxx
  ROOT/REveUtil.hxx
  ROOT/REveVSD.hxx
  ROOT/REveVSDStructs.hxx
  ROOT/REveVector.hxx
  ROOT/REveViewContext.hxx
  ROOT/REveViewer.hxx
)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTEve
  HEADERS
    ${ROOTEVE_HEADERS}
  SOURCES
    src/REveChunkManager.cxx
    src/REveBox.cxx
    src/REveBoxSet.cxx
    src/REveCalo.cxx
    src/REveCaloData.cxx
    src/REveCompound.cxx
    src/REveDataCollection.cxx
    src/REveDataProxyBuilderBase.cxx
    src/REveDataSimpleProxyBuilder.cxx
    src/REveDataTable.cxx
    src/REveDigitSet.cxx
    src/REveFrameBox.cxx
    src/REveElement.cxx
    src/REveGeoPolyShape.cxx
    src/REveGeoShape.cxx
    src/REveGeoShapeExtract.cxx
    src/REveGeoOverlaps.cxx
    src/REveGeoTopNode.cxx
    src/REveGluTess.cxx
    src/REveJetCone.cxx
    src/REveEllipsoid.cxx
    src/REveLine.cxx
    src/REveLogo.cxx
    src/REveManager.cxx
    src/REvePathMark.cxx
    src/REvePointSet.cxx
    src/REvePolygonSetProjected.cxx
    src/REveProjectionAxis.cxx
    src/REveProjectionBases.cxx
    src/REveProjectionManager.cxx
    src/REveProjections.cxx
    src/REveRenderData.cxx
    src/REveRGBAPalette.cxx
    src/REveScalableStraightLineSet.cxx
    src/REveScene.cxx
    src/REveSceneInfo.cxx
    src/REveSecondarySelectable.cxx
    src/REveSelection.cxx
    src/REveShape.cxx
    src/REveSMorph.cxx
    src/REveStraightLineSet.cxx
    src/REveTableInfo.cxx
    src/REveTableProxyBuilder.cxx
    src/REveText.cxx
    src/REveTrack.cxx
    src/REveTrackProjected.cxx
    src/REveTrackPropagator.cxx
    src/REveTrans.cxx
    src/REveTreeTools.cxx
    src/REveTypes.cxx
    src/REveUtil.cxx
    src/REveVector.cxx
    src/REveViewer.cxx
    src/REveCamera.cxx
    src/REveVSD.cxx
    src/REveVSDStructs.cxx
  DEPENDENCIES
    Core
    Geom
    Physics
    EG
    TreePlayer
    ROOTWebDisplay
    ROOTBrowserv7
    ROOTGeomViewer
  LIBRARIES
    ROOTCsg
    ${EXTRA_GL_LIBS}
  ${EXTRA_DICT_OPTS}
)

set(RELATIVE_INC_HEADERS ${ROOTEVE_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(ROOTEve
  PRIVATE
    FILE_SET private_header_files
    TYPE HEADERS
    BASE_DIRS inc/
    FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
  )

target_link_libraries(ROOTEve PUBLIC nlohmann_json::nlohmann_json)

if (APPLE AND NOT builtin_glu)
  set_source_files_properties(src/REveGluTess.cxx PROPERTIES COMPILE_DEFINITIONS "GL_SILENCE_DEPRECATION")
elseif(bultin_glu)
  set_source_files_properties(src/REveGluTess.cxx PROPERTIES COMPILE_DEFINITIONS "ROOT_BUILTIN_GLU")
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
