set (XROOFIT_HEADERS
  RooBrowser.h
  XRooFit.h
  RooFit/xRooFit/xRooFit.h
  RooFit/xRooFit/xRooNode.h
  RooFit/xRooFit/xRooNLLVar.h
  RooFit/xRooFit/xRooHypoSpace.h
  RooFit/xRooFit/xRooBrowser.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(RooFitXRooFit
  HEADERS
    ${XROOFIT_HEADERS}
  SOURCES
    src/Asymptotics.cxx
    src/xRooBrowser.cxx
    src/xRooFit.cxx
    src/xRooHypoSpace.cxx
    src/xRooNLLVar.cxx
    src/xRooNode.cxx
    src/xRooNode_interactive.cxx
  DICTIONARY_OPTIONS
    "-writeEmptyRootPCM"
  DEPENDENCIES
    HistFactory
    RooFit
    RooFitHS3
    RooStats
    FitPanel
    Gui
    Ged
    RooFitCore
    FitPanel
  LINKDEF
    inc/LinkDef.h
)

target_include_directories(RooFitXRooFit PRIVATE inc/RooFit)

set(RELATIVE_INC_HEADERS ${XROOFIT_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
if(NOT CMAKE_VERSION VERSION_LESS "3.23.0")
  target_sources(
      RooFitXRooFit
      PRIVATE
      FILE_SET private_header_files
      TYPE HEADERS
      BASE_DIRS inc/ src/
      FILES
          ${RELATIVE_INC_HEADERS}
          inc/LinkDef.h
          inc/RooFit/xRooFit/Config.h # was not part of XROOFIT_HEADERS
          src/coutCapture.h
          src/xRooFitVersion.h
  )
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
