# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT tmva package
# @author Pere Mato, CERN
############################################################################

set (EXTRA_DICT_OPTS)
if (runtime_cxxmodules AND WIN32)
  set (EXTRA_DICT_OPTS NO_CXXMODULE)
endif()

add_library(TMVAGui SHARED)
target_sources(TMVAGui
PRIVATE
  src/BDTControlPlots.cxx
  src/BoostControlPlots.cxx
  src/CorrGui.cxx
  src/CorrGuiMultiClass.cxx
  src/MovieMaker.cxx
  src/PlotFoams.cxx
  src/TMVAGui.cxx
  src/TMVAMultiClassGui.cxx
  src/TMVARegGui.cxx
  src/annconvergencetest.cxx
  src/compareanapp.cxx
  src/correlations.cxx
  src/correlationsMultiClass.cxx
  src/correlationscatters.cxx
  src/correlationscattersMultiClass.cxx
  src/deviations.cxx
  src/efficiencies.cxx
  src/efficienciesMulticlass.cxx
  src/likelihoodrefs.cxx
  src/mvas.cxx
  src/mvasMulticlass.cxx
  src/mvaweights.cxx
  src/network.cxx
  src/paracoor.cxx
  src/regression_averagedevs.cxx
  src/rulevis.cxx
  src/rulevisCorr.cxx
  src/rulevisHists.cxx
  src/tmvaglob.cxx
  src/training_history.cxx
  src/variables.cxx
  src/variablesMultiClass.cxx
  src/BDT.cxx
  src/BDT_Reg.cxx
  src/probas.cxx
  src/mvaeffs.cxx
PUBLIC
  FILE_SET HEADERS TYPE HEADERS BASE_DIRS inc/ FILES
    inc/TMVA/BDT.h
    inc/TMVA/BDTControlPlots.h
    inc/TMVA/BDT_Reg.h
    inc/TMVA/BoostControlPlots.h
    inc/TMVA/CorrGui.h
    inc/TMVA/CorrGuiMultiClass.h
    inc/TMVA/MovieMaker.h
    inc/TMVA/PlotFoams.h
    inc/TMVA/TMVAGui.h
    inc/TMVA/TMVAMultiClassGui.h
    inc/TMVA/TMVARegGui.h
    inc/TMVA/annconvergencetest.h
    inc/TMVA/compareanapp.h
    inc/TMVA/correlations.h
    inc/TMVA/correlationsMultiClass.h
    inc/TMVA/correlationscatters.h
    inc/TMVA/correlationscattersMultiClass.h
    inc/TMVA/deviations.h
    inc/TMVA/efficiencies.h
    inc/TMVA/efficienciesMulticlass.h
    inc/TMVA/likelihoodrefs.h
    inc/TMVA/mvaeffs.h
    inc/TMVA/mvas.h
    inc/TMVA/mvasMulticlass.h
    inc/TMVA/mvaweights.h
    inc/TMVA/network.h
    inc/TMVA/paracoor.h
    inc/TMVA/probas.h
    inc/TMVA/regression_averagedevs.h
    inc/TMVA/rulevis.h
    inc/TMVA/rulevisCorr.h
    inc/TMVA/rulevisHists.h
    inc/TMVA/tmvaglob.h
    inc/TMVA/training_history.h
    inc/TMVA/variables.h
    inc/TMVA/variablesMultiClass.h
  FILE_SET linkDef TYPE HEADERS BASE_DIRS inc/ FILES
    inc/LinkDef.h
)
target_link_libraries(TMVAGui PUBLIC
    RIO
    Hist
    Tree
    TreeViewer
    MLP
    XMLIO
    TMVA
    Gui
    HistPainter
    Core
)
ROOT_GENERATE_TARGET_DICTIONARY(TMVAGui -writeEmptyRootPCM ${EXTRA_DICT_OPTS})
ROOT_INSTALL_TARGET(TMVAGui)
