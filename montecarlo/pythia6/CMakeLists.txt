############################################################################
# CMakeLists.txt file for building ROOT montecarlo/pythia6 package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(montecarlo/eg)
ROOT_USE_PACKAGE(montecarlo/vmc)
ROOT_USE_PACKAGE(math/physics)

ROOT_GENERATE_DICTIONARY(G__Pythia6 *.h  MODULE EGPythia6 LINKDEF LinkDef.h)

if(pythia6_nolink)
  set(PYTHIA6_LIBRARIES)
  if(ROOT_PLATFORM STREQUAL linux)
    string(REGEX REPLACE "-Wl,--no-undefined" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
  elseif(ROOT_PLATFORM STREQUAL macosx)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")
  endif()
else()
  if(MSVC)
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /SAFESEH:NO ")
    link_directories($ENV{LIB})
  endif()
endif()

ROOT_LINKER_LIBRARY(EGPythia6 *.cxx G__Pythia6.cxx LIBRARIES Core ${PYTHIA6_LIBRARIES} DEPENDENCIES EG Graf VMC Physics)
ROOT_INSTALL_HEADERS()
