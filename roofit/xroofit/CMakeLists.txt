ROOT_STANDARD_LIBRARY_PACKAGE(RooFitXRooFit
  HEADERS
    RooBrowser.h
    XRooFit.h
    RooFit/xRooFit/xRooFit.h
    RooFit/xRooFit/xRooNode.h
    RooFit/xRooFit/xRooNLLVar.h
    RooFit/xRooFit/xRooHypoSpace.h
    RooFit/xRooFit/xRooBrowser.h
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

ROOT_ADD_TEST_SUBDIRECTORY(test)

if(webgui)
  # provider for browsing of RooWorkspace

  ROOT_LINKER_LIBRARY(ROOTFitWorkspaceProvider
      src/xRooProvider.cxx
    DEPENDENCIES
      ROOTBrowsable
      RooFitCore
      RooFitXRooFit
  )

endif(webgui)
