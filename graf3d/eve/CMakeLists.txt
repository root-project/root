############################################################################
# CMakeLists.txt file for building ROOT graf3d/eve package
############################################################################

#if(root7)
    ROOT_GLOB_SOURCES(Eve_v7_sources      ${CMAKE_CURRENT_SOURCE_DIR}/v7/src/*.cxx)
    ROOT_GLOB_HEADERS(Eve_v7_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/v7/inc/ROOT/*.hxx)
#endif()

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

if(MSVC)
  ROOT_GENERATE_DICTIONARY(G__Eve1 ${headers1} LINKDEF LinkDef1.h)
  ROOT_GENERATE_DICTIONARY(G__Eve2 ${headers2} LINKDEF LinkDef2.h)
  ROOT_GENERATE_ROOTMAP(Eve LINKDEF LinkDef.h DEPENDENCIES Geom GeomPainter Graf3d Gui Gpad Graf Hist Physics Ged EG Tree TreePlayer RGL RIO Rint MathCore)
  ROOT_LINKER_LIBRARY(Eve *.cxx G__Eve1.cxx G__Eve2.cxx LIBRARIES ${OPENGL_LIBRARIES} GLEW ${FTGL_LIBRARIES} GLEW ${FTGL_LIBRARIES} Core DEPENDENCIES Core Geom Ged RGL Physics EG)
  ROOT_INSTALL_HEADERS()
else()
  ROOT_STANDARD_LIBRARY_PACKAGE(Eve
    # HEADERS   ${headers1} ${headers2} ${Eve_v7_dict_headers}
    # SOURCES   *.cxx ${Eve_v7_sources}
    # LIBRARIES ${OPENGL_LIBRARIES} GLEW ${FTGL_LIBRARIES}
    # DEPENDENCIES Core Geom Ged RGL Physics EG)
    HEADERS   ${Eve_v7_dict_headers}
    SOURCES   ${Eve_v7_sources}
    LIBRARIES 
    DEPENDENCIES Core Geom Physics EG TreePlayer)
endif()

