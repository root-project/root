# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT rootfit/histfactory package
# @author Pere Mato, CERN
############################################################################

add_library(HistFactory SHARED)
target_sources(HistFactory
PRIVATE
  src/FlexibleInterpVar.cxx
  src/HistFactoryImpl.cxx
  src/HistFactoryModelUtils.cxx
  src/HistFactoryNavigation.cxx
  src/HistoToWorkspaceFactoryFast.cxx
  src/LinInterpVar.cxx
  src/MakeModelAndMeasurementsFast.cxx
  src/Measurement.cxx
  src/ParamHistFunc.cxx
  src/PiecewiseInterpolation.cxx
  src/RooBarlowBeestonLL.cxx
PUBLIC
  FILE_SET not4dict_header_files TYPE HEADERS BASE_DIRS inc/ FILES
    inc/HFMsgService.h # was not part of dict headers, but could well be moved to public below, or to fully private since only used by .cxx
  FILE_SET HEADERS TYPE HEADERS BASE_DIRS inc/ FILES
    inc/RooStats/HistFactory/Detail/HistFactoryImpl.h
    inc/RooStats/HistFactory/FlexibleInterpVar.h
    inc/RooStats/HistFactory/HistFactoryException.h
    inc/RooStats/HistFactory/HistFactoryModelUtils.h
    inc/RooStats/HistFactory/HistFactoryNavigation.h
    inc/RooStats/HistFactory/HistoToWorkspaceFactoryFast.h
    inc/RooStats/HistFactory/LinInterpVar.h
    inc/RooStats/HistFactory/MakeModelAndMeasurementsFast.h
    inc/RooStats/HistFactory/Measurement.h
    inc/RooStats/HistFactory/ParamHistFunc.h
    inc/RooStats/HistFactory/PiecewiseInterpolation.h
    inc/RooStats/HistFactory/RooBarlowBeestonLL.h
  FILE_SET linkDef TYPE HEADERS BASE_DIRS inc/ FILES
    inc/LinkDef.h
)
target_link_libraries(HistFactory
PUBLIC
  RooFit
  RooFitCore
  Tree
  RIO
  Hist
  MathCore
  Graf
  Gpad
  RooStats
)
# The ConfigParser can create HistFactory models from XML files. It uses TXML,
# so we won't build it if the xml build option is OFF.
if (xml)
  target_sources(HistFactory
  PRIVATE
    src/ConfigParser.cxx
  PUBLIC
  FILE_SET HEADERS TYPE HEADERS BASE_DIRS inc/ FILES
    inc/RooStats/HistFactory/ConfigParser.h
  )
  target_link_libraries(HistFactory PRIVATE XMLParser)
endif()

set (EXTRA_DICT_OPTS)
if (runtime_cxxmodules AND WIN32)
  set (EXTRA_DICT_OPTS NO_CXXMODULE)
endif()
ROOT_GENERATE_TARGET_DICTIONARY(HistFactory "-writeEmptyRootPCM" ${EXTRA_DICT_OPTS})
ROOT_INSTALL_TARGET(HistFactory)

# For recent clang, this can facilitate auto-vectorisation.
# In RooFit, the errno side effect is not needed, anyway:
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  target_compile_options(HistFactory PUBLIC -fno-math-errno)
endif()

# The hist2workspace executable uses the HistFactory::ConfigParser to read the
# XML model specification. The ConfigParser is only built when xml is ON, so we
# can't build hist2workspace without xml.
if(xml)
  ROOT_EXECUTABLE(hist2workspace hist2workspace.cxx LIBRARIES HistFactory CMAKENOEXPORT)
  target_compile_definitions(HistFactory PUBLIC HISTFACTORY_XML)
endif()

if(MSVC)
  set(prepareHistFactory_script prepareHistFactory.bat)
else()
  set(prepareHistFactory_script prepareHistFactory)
endif()
file(COPY config/${prepareHistFactory_script} DESTINATION  ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${prepareHistFactory_script}
                PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                            GROUP_EXECUTE GROUP_READ
                            WORLD_EXECUTE WORLD_READ
                DESTINATION ${CMAKE_INSTALL_BINDIR})

# workaround to include privately headers that are not public, from a "friend" target
get_target_property(RooFitCore_Friend_Dirs RooFitCore HEADER_DIRS_friend_header_files)
if(RooFitCore_Friend_Dirs)
  target_include_directories(HistFactory PRIVATE ${RooFitCore_Friend_Dirs})
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
