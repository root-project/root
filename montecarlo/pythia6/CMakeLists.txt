############################################################################
# CMakeLists.txt file for building ROOT montecarlo/pythia6 package
# @author Pere Mato, CERN
############################################################################

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

ROOT_STANDARD_LIBRARY_PACKAGE(EGPythia6
                              LIBRARIES ${PYTHIA6_LIBRARIES}
                              DEPENDENCIES Core EG Graf VMC Physics)
