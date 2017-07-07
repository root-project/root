############################################################################
# CMakeLists.txt file for building ROOT geom/vecgeom package
# @author Mihaela Gheata, CERN
############################################################################

include_directories(AFTER SYSTEM ${VECGEOM_INCLUDE_DIR})
if ( Vc_FOUND )
   include_directories( AFTER SYSTEM ${Vc_INCLUDE_DIRS})
   if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
       AND NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fabi-version=6")
   endif()
endif()
   
ROOT_GENERATE_DICTIONARY(G__ConverterVG TGeoVGConverter.h TGeoVGShape.h MODULE ConverterVG LINKDEF LinkDef.h DEPENDENCIES Geom)

ROOT_LINKER_LIBRARY(ConverterVG *.cxx G__ConverterVG.cxx LIBRARIES ${VECGEOM_LIBRARIES} -ldl DEPENDENCIES Geom)

ROOT_INSTALL_HEADERS()
