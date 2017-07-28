############################################################################
# CMakeLists.txt file for building ROOT graf3d/eve package
############################################################################

include_directories(${OPENGL_INCLUDE_DIR})

set(headers1 TEveBrowser.h TEveChunkManager.h TEveCompound.h
             TEveElement*.h TEveEventManager.h TEveGValuators.h
             TEveGedEditor.h TEveMacro.h TEveManager.h TEvePad.h TEveParamList.h
             TEveProjectionAxes*.h TEveProjectionBases.h TEveProjectionManager*.h
             TEveProjections.h TEveScene*.h TEveSelection.h TEveTrans*.h TEveTreeTools.h
             TEveUtil.h TEveVector.h TEvePathMark.h TEveVSD.h TEveViewer*.h TEveWindow*.h
             TEveSecondarySelectable.h)
set(headers2 TEveArrow*.h TEveBox*.h TEveCalo*.h
             TEveDigitSet*.h TEveFrameBox*.h TEveGeo*.h
             TEveGridStepper*.h TEveLegoEventHandler*.h TEveShape*.h
             TEveLine*.h TEvePointSet*.h TEvePolygonSetProjected*.h TEveQuadSet*.h
             TEveRGBAPalette*.h TEveScalableStraightLineSet*.h TEveStraightLineSet*.h
             TEveText*.h TEveTrack*.h TEveTriangleSet*.h TEveJetCone*.h
             TEvePlot3D*.h)

if(MACOSX_GLU_DEPRECATED)
  add_definitions(-Wno-deprecated-declarations)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(Eve
                              HEADERS ${headers1} ${headers2}
                              LIBRARIES ${OPENGL_LIBRARIES} GLEW ${FTGL_LIBRARIES}
                              DEPENDENCIES Core Geom Ged RGL Physics EG)
