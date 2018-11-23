############################################################################
# CMakeLists.txt file for building ROOT geom/geocad package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(GeoCad
  HEADERS
    TGeoToOCC.h
    TGeoToStep.h
    TOCCToStep.h
  SOURCES
    TGeoToOCC.cxx
    TGeoToStep.cxx
    TOCCToStep.cxx
  DICTIONARY_OPTIONS
    -I${OCC_INCLUDE_DIR}
  DEPENDENCIES
    Geom
)

ROOT_ADD_CXX_FLAG(_GeoCad_FLAGS -Wno-extra)
ROOT_ADD_CXX_FLAG(_GeoCad_FLAGS -Wno-ignored-qualifiers)
ROOT_ADD_CXX_FLAG(_GeoCad_FLAGS -Wno-overloaded-virtual)
ROOT_ADD_CXX_FLAG(_GeoCad_FLAGS -Wno-shadow)
separate_arguments(_GeoCad_FLAGS)

target_compile_options(GeoCad PRIVATE ${_GeoCad_FLAGS})
target_link_libraries(GeoCad PUBLIC ${OCC_LIBRARIES})
target_include_directories(GeoCad PUBLIC ${OCC_INCLUDE_DIR})
