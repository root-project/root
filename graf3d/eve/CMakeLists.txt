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

# We need to disallow the direct use of gl.h. This way people will see the error
# and the suggested fix. This happens by providing our own "fake" system gl.h.
set(CMAKE_CXX_FLAGS " -isystem ${CMAKE_CURRENT_SOURCE_DIR}/../glew/isystem ${CMAKE_CXX_FLAGS}")

ROOT_GENERATE_DICTIONARY(G__Eve ${headers1} ${headers2} MODULE Eve LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(Eve *.cxx G__Eve.cxx LIBRARIES ${OPENGL_LIBRARIES} GLEW ${FTGL_LIBRARIES} Core DEPENDENCIES Geom Ged RGL Physics EG)

ROOT_INSTALL_HEADERS()
