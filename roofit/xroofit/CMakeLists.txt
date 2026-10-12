if(CMAKE_VERSION VERSION_LESS "3.23.0")
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
      src/PythonInterface.cxx
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
else()
  add_library(RooFitXRooFit SHARED)
  target_sources(
    RooFitXRooFit
    PRIVATE
      src/Asymptotics.cxx
      src/xRooBrowser.cxx
      src/xRooFit.cxx
      src/xRooHypoSpace.cxx
      src/xRooNLLVar.cxx
      src/xRooNode.cxx
      src/xRooNode_interactive.cxx
      src/PythonInterface.cxx
      # These below could be also a PRIVATE header file_set if wanted
      src/PythonInterface.h
      src/coutCapture.h
      src/xRooFitVersion.h
    PUBLIC
      FILE_SET HEADERS
        TYPE HEADERS
        BASE_DIRS inc/RooFit
        FILES
          inc/RooFit/xRooFit/xRooFit.h
          inc/RooFit/xRooFit/xRooNode.h
          inc/RooFit/xRooFit/xRooNLLVar.h
          inc/RooFit/xRooFit/xRooHypoSpace.h
          inc/RooFit/xRooFit/xRooBrowser.h
          inc/RooFit/xRooFit/Config.h # was not part of XROOFIT_HEADERS earlier
    PUBLIC
      FILE_SET extraHEADERS
        TYPE HEADERS
        BASE_DIRS inc/
        FILES
          inc/RooBrowser.h
          inc/XRooFit.h
      FILE_SET linkDefs
        TYPE HEADERS
        BASE_DIRS inc/
        FILES
          inc/LinkDef.h
  )
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
  install(TARGETS RooFitXRooFit
    EXPORT ${CMAKE_PROJECT_NAME}Exports
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT applications

    # with this we no longer need the moveheader target etc
    FILE_SET HEADERS
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # add /ROOT to make Debian happy

    FILE_SET extraHEADERS
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # add /ROOT to make Debian happy

    FILE_SET linkDefs
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # add /ROOT to make Debian happy
  )
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
