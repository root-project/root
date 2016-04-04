############################################################################
# CMakeLists.txt file for building ROOT geom/vecgeom package
# @author Mihaela Gheata, CERN
############################################################################

include_directories(AFTER SYSTEM ${VECGEOM_INCLUDE_DIR})
if ( Vc_FOUND )
   include_directories( AFTER SYSTEM ${Vc_INCLUDE_DIRS})
   set(VC_LIBRARIES ${Vc_LIBRARIES})
   if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=6")
   endif()
endif()
   
ROOT_GENERATE_DICTIONARY(G__ConverterVG TGeoVGConverter.h MODULE ConverterVG LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(ConverterVG *.cxx G__ConverterVG.cxx LIBRARIES Core ${VECGEOM_LIBRARIES} -ldl DEPENDENCIES Geom)

ROOT_INSTALL_HEADERS()
