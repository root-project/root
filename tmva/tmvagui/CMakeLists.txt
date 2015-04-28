############################################################################
# CMakeLists.txt file for building ROOT tmva package
# @author Pere Mato, CERN
############################################################################
#---The following is needed becuase client codes of TMVA (in ROOT itself) assumes to find header files under /TVMA
#execute_process(COMMAND cmake -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/inc/TMVA)

if(NOT CMAKE_PROJECT_NAME STREQUAL ROOT)
  cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
  project(TMVA)
  include(${ROOTSYS}/cmake/modules/StandaloneBuild.cmake)
endif()

ROOT_USE_PACKAGE(tmva/tmva) 
ROOT_USE_PACKAGE(gui/gui) 

set(headers1 annconvergencetest.h  deviations.h mvaeffs.h PlotFoams.h  TMVAGui.h
	 BDTControlPlots.h  correlationscatters.h efficiencies.h  mvas.h probas.h 
	 BDT.h   correlationscattersMultiClass.h  likelihoodrefs.h  mvasMulticlass.h  regression_averagedevs.h  TMVAMultiClassGui.h
	 BDT_Reg.h  correlations.h   mvaweights.h rulevisCorr.h  TMVARegGui.h
	 BoostControlPlots.h correlationsMultiClass.h network.h rulevis.h   variables.h
	 CorrGui.h  paracoor.h  rulevisHists.h variablesMultiClass.h
	 compareanapp.h  CorrGuiMultiClass.h   MovieMaker.h tmvaglob.h)

#---Need to suffix each header name by TMVA/  -----------------
foreach(hs headers1)
  foreach(h ${${hs}})
    set(t${hs} ${t${hs}} TMVA/${h})
  endforeach()
endforeach()

ROOT_GENERATE_DICTIONARY(G__TMVAGui ${theaders1}   MODULE TMVAGui LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")

ROOT_LINKER_LIBRARY(TMVAGui *.cxx G__TMVAGui.cxx LIBRARIES Core
                    DEPENDENCIES RIO Hist Tree TreeViewer  MLP Minuit XMLIO TMVA Gui)

install(DIRECTORY inc/TMVA/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/TMVA
                            COMPONENT headers
                            PATTERN ".svn" EXCLUDE
                            REGEX "LinkDef" EXCLUDE )
set_property(GLOBAL APPEND PROPERTY ROOT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/inc)

if(NOT gnuinstall)
  install(DIRECTORY test DESTINATION tmva 
  FILES_MATCHING PATTERN "*.C" 
                 PATTERN "*.gif" 
                 PATTERN "*.png" 
                 PATTERN "README"
                 PATTERN "data" EXCLUDE)
endif()

#ROOT_ADD_TEST_SUBDIRECTORY(test)




