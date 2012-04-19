############################################################################
# CMakeLists.txt file for building ROOT graf3d/eve package
############################################################################

ROOT_USE_PACKAGE(gui/ged)
ROOT_USE_PACKAGE(graf3d/gl)
ROOT_USE_PACKAGE(geom/geom)
ROOT_USE_PACKAGE(math/physics)
ROOT_USE_PACKAGE(montecarlo/eg)
ROOT_USE_PACKAGE(graf3d/glew)

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

ROOT_GENERATE_DICTIONARY(G__Eve1 ${headers1} LINKDEF src/SolarisCCDictHack.h LinkDef1.h)
ROOT_GENERATE_DICTIONARY(G__Eve2 ${headers2} LINKDEF src/SolarisCCDictHack.h LinkDef2.h)

ROOT_GENERATE_ROOTMAP(Eve LINKDEF LinkDef1.h LinkDef2.h DEPENDENCIES Geom GeomPainter Graf3d Gui Gpad Graf Hist Physics Ged EG Tree TreePlayer RGL RIO Rint MathCore)
ROOT_LINKER_LIBRARY(Eve *.cxx G__Eve1.cxx G__Eve2.cxx LIBRARIES ${OPENGL_LIBRARIES} GLEW FTGL Cint Core DEPENDENCIES Geom Ged RGL Physics EG)

ROOT_INSTALL_HEADERS()