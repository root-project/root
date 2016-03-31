############################################################################
# CMakeLists.txt file for building ROOT geom/vecgeom package
# @author Mihaela Gheata, CERN
############################################################################

include_directories(AFTER SYSTEM ${VECGEOM_INCLUDE_DIR})
if ( Vc_FOUND )
   include_directories( AFTER SYSTEM ${Vc_INCLUDE_DIRS})
   set(VC_LIBRARIES ${Vc_LIBRARIES})
endif()
   
ROOT_GENERATE_DICTIONARY(G__VecGeom TGeoVGConverter.h MODULE VecGeom LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(VecGeom *.cxx G__VecGeom.cxx LIBRARIES Core ${VECGEOM_LIBRARIES} DEPENDENCIES Geom)

ROOT_INSTALL_HEADERS()
