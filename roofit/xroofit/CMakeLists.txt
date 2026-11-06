add_library(RooFitXRooFit SHARED)
target_sources(RooFitXRooFit
PRIVATE
  src/Asymptotics.cxx
  src/xRooBrowser.cxx
  src/xRooFit.cxx
  src/xRooHypoSpace.cxx
  src/xRooNLLVar.cxx
  src/xRooNode.cxx
  src/xRooNode_interactive.cxx
  src/PythonInterface.cxx
PRIVATE
  FILE_SET privateHEADERS TYPE HEADERS BASE_DIRS src/ FILES
    src/PythonInterface.h
    src/coutCapture.h
    src/xRooFitVersion.h
PUBLIC
  FILE_SET HEADERS TYPE HEADERS BASE_DIRS inc/ FILES
    inc/RooFit/xRooFit/xRooFit.h
    inc/RooFit/xRooFit/xRooNode.h
    inc/RooFit/xRooFit/xRooNLLVar.h
    inc/RooFit/xRooFit/xRooHypoSpace.h
    inc/RooFit/xRooFit/xRooBrowser.h
    inc/RooFit/xRooFit/Config.h
    inc/RooBrowser.h
    inc/XRooFit.h
  FILE_SET linkDef TYPE HEADERS BASE_DIRS inc/ FILES
    inc/LinkDef.h
)
target_include_directories(RooFitXRooFit PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/RooFit>) # for easening life in .cxx
target_link_libraries(RooFitXRooFit
PUBLIC
  HistFactory
  RooFit
  RooFitHS3
  RooStats
  FitPanel
  Gui
  Ged
  RooFitCore
  FitPanel
)

ROOT_GENERATE_TARGET_DICTIONARY(RooFitXRooFit "-writeEmptyRootPCM")
ROOT_INSTALL_TARGET(RooFitXRooFit)

ROOT_ADD_TEST_SUBDIRECTORY(test)
