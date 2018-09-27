############################################################################
# CMakeLists.txt file for building ROOT net/bonjour package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Bonjour
  HEADERS
    TBonjourBrowser.h
    TBonjourRecord.h
    TBonjourRegistrar.h
    TBonjourResolver.h
  SOURCES
    TBonjourBrowser.cxx
    TBonjourRecord.cxx
    TBonjourRegistrar.cxx
    TBonjourResolver.cxx
  DEPENDENCIES
    Core
)

target_link_libraries(Bonjour PUBLIC ${BONJOUR_LIBRARIES})
target_include_directories(Bonjour PUBLIC ${BONJOUR_INCLUDE_DIRS})
