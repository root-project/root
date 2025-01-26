# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# CMakeLists.txt for the ROOT tutorials programs.
# Author: Pere Mato, 25/10/2010
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(tutorials)

# Sergey: make no sence while CMakeLists.txt file cannot be used separately from ROOT
# but variables like ROOT_asimage_FOUND used here and produced in ROOTConfig.cmake
find_package(ROOT REQUIRED)

if(DEFINED ROOT_SOURCE_DIR)  # Testing using the binary tree
  set(ROOT_root_CMD root.exe)
  if(NOT MSVC)  # Ignore environment on Windows
    set(ROOT_environ PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH}
                     ${ld_library_path}=${CMAKE_BINARY_DIR}/lib:$ENV{${ld_library_path}}
                     ROOTSYS=${CMAKE_BINARY_DIR}
                     PYTHONPATH=${CMAKE_BINARY_DIR}/lib:$ENV{PYTHONPATH})
  else()
    set(ROOT_environ ROOTSYS=${CMAKE_BINARY_DIR}
                     PYTHONPATH=${CMAKE_BINARY_DIR}/bin;$ENV{PYTHONPATH})
  endif()
else()                       # testing using an installation
  include(${ROOT_USE_FILE})
  if(DEFINED ROOT_CONFIG_EXECUTABLE) #---If ROOT was built with the classic configure/make---
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules)
    include(RootMacros)
    set(ROOT_root_CMD root.exe)
  endif()
  enable_testing()
endif()

# Set the environment for the tutorials, which is the eventual ROOT_environ
# plus some environment variables related to limiting the number of threads
# used by NumPy.
# See: https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
set(TUTORIAL_ENV ${ROOT_environ} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1)

#---Copy the CTestCustom.cmake file into the build directory--------
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

#---Provide a rootlogon.C file in the current build directory that
#   will affect the way we run all tutorials.
#   This overwrites the existing rootlogon.C and rootalias.C in the
#   tutorials directory which is copied to the build area.
#-------------------------------------------------------------------
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/rootlogon.C "{
  // Needed by ACLiC to use the current directory for scratch area
  gSystem->SetBuildDir(\".\", kTRUE);
}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/rootalias.C "")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/rootlogoff.C "{}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/.rootrc "
Proof.Sandbox: /tmp/proof
Rint.History:  .root_hist
ACLiC.LinkLibs:  1
")

#---Tutorials that need substantial network to work------------------
set(need_network analysis/dataframe/df027_SQliteDependencyOverVersion.C)

#---Tutorials disabled depending on the build components-------------

if(NOT clad)
  set(clad_veto math/fit/minuit2GausFit.C
                # TODO: the next veto should not be necessary!
                # The tutorial should also work without clad.
                # See https://github.com/root-project/root/issues/15091.
                math/fit/exampleFit3D.C
  )
endif()

if(MSVC AND NOT win_broken_tests)
  # RBatchGenerator tutorials don't work on Windows at the moment.
  list(APPEND dataframe_veto machine_learning/RBatchGenerator_NumPy.py)
  list(APPEND dataframe_veto machine_learning/RBatchGenerator_TensorFlow.py)
  list(APPEND dataframe_veto machine_learning/RBatchGenerator_PyTorch.py)
  list(APPEND dataframe_veto machine_learning/RBatchGenerator_filters_vectors.py)
  # df036* and df037* seem to trigger OS errors when trying to delete the
  # test files created in the tutorials. It is unclear why.
  list(APPEND dataframe_veto analysis/dataframe/df036_missingBranches.C)
  list(APPEND dataframe_veto analysis/dataframe/df036_missingBranches.py)
  list(APPEND dataframe_veto analysis/dataframe/df037_TTreeEventMatching.C)
  list(APPEND dataframe_veto analysis/dataframe/df037_TTreeEventMatching.py)
  # The RooFit SBI tutorials fail on Windows for unknown reasons
  list(APPEND roofit_veto roofit/roofit/rf617_simulation_based_inference_multidimensional.py)
endif()

# TODO: fix the problem and re-enable the tutorial test. The rf615 tutorial
# occasionally fails on cleanup on different platforms, hinting to a PyROOT
# issue. We disable the rf617 tutorial for now. as the covered RooFit
# functionality is also covered by rf617 (the multidimensional case).
list(APPEND roofit_veto roofit/roofit/rf615_simulation_based_inference.py)

if (NOT dataframe)
    # RDataFrame
    list(APPEND dataframe_veto analysis/dataframe/*.C analysis/dataframe/*.py)
    # RDataFrame tutorial in graphs
    list(APPEND dataframe_veto visualisation/graphs/gr109_timeSeriesFromCSV_RDF.C)
    # TMVA tutorials dependent on RDataFrame
    list(APPEND dataframe_veto machine_learning/tmva*.C)
    list(APPEND dataframe_veto machine_learning/TMVA_SOFIE_RDataFrame*.C)
    list(APPEND dataframe_veto machine_learning/TMVA_SOFIE_RDataFrame*.py)
    list(APPEND dataframe_veto machine_learning/TMVA_SOFIE_Inference.py)
    list(APPEND dataframe_veto machine_learning/RBatchGenerator_NumPy.py)
    list(APPEND dataframe_veto machine_learning/RBatchGenerator_TensorFlow.py)
    list(APPEND dataframe_veto machine_learning/RBatchGenerator_PyTorch.py)
    list(APPEND dataframe_veto machine_learning/RBatchGenerator_filters_vectors.py)
    # RooFit tutorial depending on RDataFrame
    list(APPEND dataframe_veto roofit/roofit/rf408*)
endif()

if(NOT sqlite)
    # RDF+SQlite tutorials
    list(APPEND dataframe_veto analysis/dataframe/*SQlite*)
endif()
if(NOT davix)
    list(APPEND dataframe_veto analysis/dataframe/df027_SQliteDependencyOverVersion.C)
    list(APPEND dataframe_veto analysis/dataframe/df028_SQliteIPLocation.C)
    list(APPEND dataframe_veto analysis/dataframe/df029_SQlitePlatformDistribution.C)
    list(APPEND dataframe_veto analysis/dataframe/df030_SQliteVersionsOfROOT.C)
endif()

if(MACOSX_VERSION VERSION_EQUAL 10.13)
   list(APPEND dataframe_veto analysis/dataframe/df103_NanoAODHiggsAnalysis.*)
endif()

if(NOT geom)
  set(geom_veto visualisation/geom/*.C geom/*.C visualisation/geom/gdml/*.C)
endif()

if(NOT ROOT_proof_FOUND)
   set(proof_veto roofit/roostats/StandardFrequentistDiscovery.C)
endif()

if(NOT ROOT_spectrum_FOUND)
   set(spectrum_veto legacy/spectrum/*.C)
endif()

if(NOT ROOT_roofit_FOUND)
  set(roofit_veto roofit/roofit/*.C roofit/roofit/*.py
                  roofit/roostats/*.C roofit/roostats/*.py
                  roofit/histfactory/*.C roofit/histfactory/*.py)
else()
  if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 4 AND NOT llvm13_broken_tests)
    # The following tutorials are failing with this error:
    # IncrementalExecutor::executeFunction: symbol '__std_find_trivial_4@12' unresolved while linking [cling interface function]!
    # on Windows 32 bit and Visual Studio v17.8
    list(APPEND roofit_veto roofit/roofit/rf614_binned_fit_problems.C)
  endif()
endif()

if(NOT ROOT_unuran_FOUND)
  set(unuran_veto  math/testrandom.C math/unuran/unuranDemo.C math/unuran/unuranFoamTest.C
                   math/multidimSampling.C)
endif()

if(NOT ROOT_xml_FOUND)
  set(xml_veto  io/xml/*.C
                roofit/histfactory/*.C   # histfactory requires xml
                roofit/histfactory/*.py
                roofit/roostats/*.C      # most roostats tutorials require xml too,
                roofit/roostats/*.py)    # because they create test data with histfactory
endif()

if(NOT ROOT_unfold_FOUND)
  list(APPEND xml_veto analysis/unfold/*.C)
endif()

if(NOT ROOT_mpi_FOUND)
  set(mpi_veto io/testTMPIFile.C)
endif()

if(NOT xrootd)
  set(xrootd_veto analysis/dataframe/df101_h1Analysis.C
                  analysis/dataframe/df102_NanoAODDimuonAnalysis.C
                  analysis/dataframe/df103_NanoAODHiggsAnalysis.C
                  analysis/dataframe/df106_HiggsToFourLeptons.C
                  machine_learning/tmva103_Application.C
                  analysis/dataframe/df033_Describe.py
                  analysis/dataframe/df102_NanoAODDimuonAnalysis.py
                  analysis/dataframe/df103_NanoAODHiggsAnalysis.py
                  analysis/dataframe/df104_HiggsToTwoPhotons.py
                  analysis/dataframe/df105_WBosonAnalysis.py
                  analysis/dataframe/df106_HiggsToFourLeptons.py
                  analysis/dataframe/df107_SingleTopAnalysis.py
                  roofit/roofit/rf618_mixture_models.py # depends on df106_HiggsToFourLeptons.py
                  experimental/rcanvas/df104.py
                  experimental/rcanvas/df105.py
                  )
endif()

# variables identifying the package must have the package name  in lower case (it corresponds to the CMake option name)
if(NOT ROOT_r_FOUND)
  set(r_veto  math/r/*.C)
endif()

set(histfactory_veto roofit/histfactory/makeExample.C)

if(NOT ROOT_fitsio_FOUND)
  set(fitsio_veto  io/fitsio/*.C)
endif()

if(NOT ROOT_mathmore_FOUND)
  set(mathmore_veto
      math/quasirandom.C
      math/exampleMultiRoot.C
      math/pdf/pdf009_Bessel.C
      math/LegendreAssoc.C
      math/Legendre.C
      math/mathmoreIntegration.C
      math/pdf/pdf007_multivarGaus.C
      math/pdf/pdf012_tStudent.C
      math/pdf/pdf001_Nornal.C
      roofit/roostats/TestNonCentral.C
      roofit/roostats/TestNonCentral.py
      math/Legendre.py
      math/pdf/pdf009_Bessel.py
      math/pdf/pdf012_tStudent.py)
endif()

if(NOT ROOT_fftw3_FOUND)
  set(fftw3_veto roofit/roofit/rf208_convolution.C
                 roofit/roofit/rf210_angularconv.C
                 roofit/roofit/rf211_paramconv.C
                 roofit/roofit/rf512_wsfactory_oper.C
                 roofit/roofit/rf208_convolution.py
                 roofit/roofit/rf210_angularconv.py
                 roofit/roofit/rf211_paramconv.py
                 roofit/roofit/rf512_wsfactory_oper.py
                 math/fft/FFT.C
                 math/fit/fitConvolution.C
                 math/fit/fitConvolution.py)
endif()

if(NOT ROOT_opengl_FOUND)
  set(opengl_veto io/tree/tree502_staff.C
                  visualisation/gl/*.C)
endif()

if(NOT GRAPHVIZ_FOUND)
  set(gviz_veto visualisation/graphs/gr016_struct.C)
endif()

if(NOT TBB_FOUND AND NOT builtin_tbb)
  set(tbb_veto  analysis/mtbb*.C io/tree/mtbb*.C)
endif()

if(NOT ROOT_imt_FOUND)
  set(imt_veto io/tree/imt*.C io/tree/mt*.C analysis/mt*.C)
endif()
if(MSVC)
  #---Multiproc is not supported on Windows
  set(imt_veto ${imt_veto} analysis/mp*.C io/tree/mp*.C legacy/multicore/mp*.C  multicore/mp*.C ./analysis/mtbb_parallelHistoFill.C)
endif()

if(ROOT_CLASSIC_BUILD)
  set(classic_veto legacy/multicore/mp_*.C analysis/mp_*.C io/tree/mp_*.C)
endif()

if(NOT gdml)
  set(gdml_veto visualisation/geom/gdml/testoptical.C)
endif()

#---These ones requires a display to run-----------------------------
set(gui_veto math/fit/fitpanel_playback.C
             visualisation/cocoa/*.C
             visualisation/geom/building.C visualisation/geom/cheongwadae.C visualisation/geom/geom*.C visualisation/geom/lego.C visualisation/geom/robot.C visualisation/geom/south_gate.C visualisation/geom/station*.C visualisation/geom/tank.C visualisation/geom/webdemo.C visualisation/geom/web_cms.cxx
             visualisation/gl/glViewerExercise.C visualisation/gl/glViewerLOD.C visualisation/gl/gviz3d.C visualisation/gl/nucleus.C visualisation/gl/viewer3DLocal.C visualisation/gl/viewer3DMaster.C
             visualisation/gui/*.C
             hist/hist057_TExec_th1.C
             hist/hist058_TExec_th2.C
             hist/hist041_TProfile2Poly_realistic.C
             hist/hist042_TProfile2Poly_module_error.C
             visualisation/image/*.C
             visualisation/graphics/psview.C visualisation/graphics/gtime.C
             visualisation/graphics/graph_edit_playback.C
             roofit/roostats/ModelInspector.C
             roofit/roostats/ModelInspector.py
             legacy/tree/tvdemo.C
             visualisation/eve/*.C
             visualisation/webgui/geom/geom_threejs.cxx
             visualisation/webgui/panel/webpanel.cxx
             visualisation/webgui/webwindow/webwindow.cxx)
if (NOT webgui)
  list(APPEND gui_veto visualisation/graphics/save_batch.C experimental/rcanvas/df104.py experimental/rcanvas/df105.py)
endif()

if (NOT ROOT_tmva_FOUND)
  list(APPEND tmva_veto machine_learning/*.C machine_learning/*.py machine_learning/envelope/*.C machine_learning/keras/*.C machine_learning/keras/*.py machine_learning/pytorch/*.py )
else()
  #---These do not need to run for TMVA
  list(APPEND tmva_veto machine_learning/createData.C)
  if(MSVC AND NOT win_broken_tests)
    list(APPEND tmva_veto machine_learning/envelope/classification.C)
  endif()
  #these depends on external packages
  if (machine_learning-pymva)
    ROOT_FIND_PYTHON_MODULE(torch QUIET)
    ROOT_FIND_PYTHON_MODULE(keras QUIET)
    ROOT_FIND_PYTHON_MODULE(sonnet QUIET)
    ROOT_FIND_PYTHON_MODULE(graph_nets QUIET)
    ROOT_FIND_PYTHON_MODULE(sklearn QUIET)
  endif()
  if (NOT ROOT_KERAS_FOUND)
    list(APPEND tmva_veto machine_learning/TMVA_SOFIE_Keras.C)
    list(APPEND tmva_veto machine_learning/TMVA_SOFIE_Keras_HiggsModel.C)
    list(APPEND tmva_veto machine_learning/TMVA_SOFIE_RDataFrame.C)
    list(APPEND tmva_veto machine_learning/TMVA_SOFIE_RDataFrame.py)
    list(APPEND tmva_veto machine_learning/TMVA_SOFIE_RDataFrame_JIT.C)
    list(APPEND tmva_veto machine_learning/TMVA_SOFIE_Models.py)
    list(APPEND tmva_veto machine_learning/TMVA_SOFIE_Inference.py)
    list(APPEND tmva_veto machine_learning/TMVA_SOFIE_RSofieReader.C)
    list(APPEND tmva_veto machine_learning/RBatchGenerator_TensorFlow.py)
  endif()
  if (NOT ROOT_SKLEARN_FOUND)
    list(APPEND tmva_veto machine_learning/TMVA_SOFIE_Models.py roofit/roofit/rf617_simulation_based_inference_multidimensional.py)
  endif()
  if (NOT ROOT_TORCH_FOUND)
    list(APPEND tmva_veto machine_learning/TMVA_SOFIE_PyTorch.C)
    list(APPEND tmva_veto machine_learning/RBatchGenerator_PyTorch.py)
  endif()
  #veto this tutorial since it is added directly
  list(APPEND tmva_veto machine_learning/TMVA_SOFIE_GNN_Parser.py)
  if (NOT ROOT_SONNET_FOUND OR NOT ROOT_GRAPH_NETS_FOUND)
    list(APPEND tmva_veto machine_learning/TMVA_SOFIE_GNN.py)
    list(APPEND tmva_veto machine_learning/TMVA_SOFIE_GNN_Application.C)
  endif()
  if (NOT machine_learning-sofie)
    list(APPEND tmva_veto machine_learning/TMVA_SOFIE_ONNX.C)
  else()
    #copy ONNX file needed for the tutorial
    configure_file(${CMAKE_SOURCE_DIR}/machine_learning/sofie/test/input_models/Linear_16.onnx ${CMAKE_BINARY_DIR}/tutorials/machine_learning/Linear_16.onnx COPYONLY)
  endif()

endif()

if (NOT ROOT_pythia8_FOUND)
  set(pythia_veto evegen/evegen_000_pythia8.C)
else()
  if("$ENV{PYTHIA8}" STREQUAL "")
    get_filename_component(pythia8dir "${PYTHIA8_INCLUDE_DIR}" DIRECTORY)
    list(APPEND TUTORIAL_ENV PYTHIA8=${pythia8dir})
  endif()
  if("$ENV{PYTHIA8DATA}" STREQUAL "" AND PYTHIA8_DATA)
    list(APPEND TUTORIAL_ENV PYTHIA8DATA=${PYTHIA8_DATA})
  endif()
endif()

if (NOT ROOT_vecgeom_FOUND)
  set(vecgeom_veto visualisation/geom/tessellatedNav.C)
endif()

if(root7)
  set(root7_veto analysis/dataframe/df013_InspectAnalysis.C
                 experimental/browser.cxx
                 experimental/filedialog.cxx
                 experimental/fitpanel.cxx
                 experimental/fitpanel6.cxx
      )
  if(NOT davix)
    list(APPEND root7_veto io/ntuple/ntpl004_dimuon.C)
    list(APPEND root7_veto io/ntuple/ntpl011_global_temperatures.C)
  endif()
  if(NOT dataframe)
    list(APPEND root7_veto io/ntuple/ntpl011_global_temperatures.C)
    list(APPEND root7_veto io/ntuple/ntpl004_dimuon.C)
    list(APPEND root7_veto io/ntuple/ntpl008_import.C)
    list(APPEND root7_veto experimental/rcanvas/df104.py)
    list(APPEND root7_veto experimental/rcanvas/df105.py)
  endif()
  if(MSVC AND NOT win_broken_tests)
    #---EOS is not supported on Windows
    list(APPEND root7_veto experimental/rcanvas/df104.py)
    list(APPEND root7_veto experimental/rcanvas/df105.py)
    list(APPEND root7_veto experimental/rcanvas/rbox.py)
  endif()
else()
  if(MSVC AND NOT win_broken_tests)
    list(APPEND root7_veto analysis/dataframe/df013_InspectAnalysis.C)
  endif()
  file(GLOB v7_veto_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/ experimental/*.py experimental/*.cxx experimental/*/*.cxx experimental/*.C experimental/*/*.C experimental/rcanvas/*.py experimental/rcanvas/*.cxx)
  list(APPEND root7_veto ${v7_veto_files})
endif()

if (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES arm64)
   set(macm1_veto analysis/dataframe/df107_SingleTopAnalysis.py)
endif()

#---These ones are disabled !!! ------------------------------------
set(extra_veto
  legacy/benchmarks.C
  legacy/htmlex.C
  legacy/rootalias.C          # Helper macro
  rootlogon.C          # Helper macro
  rootlogoff.C         # Helper macro
  legacy/rootmarks.C          # Instrumentation. Not a standalone tutorial
  legacy/multicore/mp_H1_lambdas.C # not a tutorial; used by mp104_processH1.C et al.
  html/*.C
  legacy/net/*.C
  legacy/proof/*.C
  io/sql/*.C
  io/tree/hsimpleProxy.C # A driver uses this macro which cannot be executed directly
  io/tree/tree103_tree.C
  io/tree/tree106_tree.C
  io/tree/tree108_tree.C
  roofit/roostats/rs401d_FeldmanCousins.C  # Takes too much time
  roofit/roostats/rs401d_FeldmanCousins.py
  roofit/histfactory/ModifyInterpolation.C
  io/tree/tree110_copy.C
  io/tree/tree111_copy.C
  io/tree/tree112_copy.C
  analysis/tree/h1analysis*.C # these are not a tutorial but classes used in run_h1analysis.C
  analysis/tree/h1chain.C
  http/*.C
  visualisation/eve7/*.C
  math/r/rootlogon.C)

if(MSVC)
  # disable run_h1analysis.C because of Endpoint Security HTTP traffic scanning,
  # which is corrupting the data on Windows
  list(APPEND extra_veto analysis/tree/run_h1analysis.C)
endif()

if(MSVC AND NOT llvm13_broken_tests)
  list(APPEND extra_veto
       math/exampleFunction.py
       analysis/dataframe/df002_dataModel.C
       analysis/dataframe/df016_vecOps.C
       analysis/dataframe/df017_vecOpsHEP.C
       analysis/dataframe/df002_dataModel.py
       analysis/dataframe/df016_vecOps.py
       analysis/dataframe/df017_vecOpsHEP.py
       analysis/dataframe/df032_RDFFromNumpy.py
       analysis/dataframe/df035_RDFFromPandas.py
       analysis/unfold/testUnfold7c.C)
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    list(APPEND extra_veto
         analysis/dataframe/df007_snapshot.C
         visualisation/graphics/earth.C
         visualisation/graphs/gr015_smooth.C
         io/ntuple/ntpl001_staff.C)
  endif()
endif()

set(all_veto hsimple.C
             visualisation/geom/geometry.C
             ${extra_veto}
             ${gdml_veto}
             ${gui_veto}
             ${roofit_veto}
             ${unuran_veto}
             ${xml_veto}
             ${mpi_veto}
             ${fitsio_veto}
             ${tmva_veto}
             ${mathmore_veto}
             ${fftw3_veto}
             ${opengl_veto}
             ${gviz_veto}
             ${r_veto}
             ${runtime_cxxmodules_veto}
             ${histfactory_veto}
             ${tbb_veto}
             ${imt_veto}
             ${classic_veto}
             ${geom_veto}
             ${pythia_veto}
             ${vecgeom_veto}
             ${root7_veto}
             ${xrootd_veto}
             ${spectrum_veto}
             ${dataframe_veto}
             ${macm1_veto}
             ${clad_veto}
             ${proof_veto}
             )

file(GLOB_RECURSE tutorials RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.C)
if(webgui)
  file(GLOB_RECURSE visualisation/tutorials_webcanv RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} visualisation/webcanv/*.cxx)
  list(APPEND tutorials ${visualisation/tutorials_webcanv})
endif()
if(root7 AND webgui)
  file(GLOB_RECURSE tutorials_exp RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} experimental/*.cxx)
  list(APPEND tutorials ${tutorials_exp})
endif()
file(GLOB tutorials_veto RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${all_veto})

list(LENGTH tutorials nTotal)
list(REMOVE_ITEM tutorials ${tutorials_veto})
list(LENGTH tutorials nAfterVeto)
message(STATUS "${nAfterVeto}/${nTotal} C++ tutorials have been activated.")


if(mpi)
  set (temp_list ${tutorials})
  list(FILTER tutorials INCLUDE REGEX "MPI")
  set(mpi_tutorials ${tutorials})
  set(tutorials ${temp_list})
  list(REMOVE_ITEM tutorials ${mpi_tutorials})
endif()

#---Special return code------------------------------------------------
set(returncode_1 math/fit/fit2a.C
                 visualisation/graphics/earth.C
                 visualisation/graphics/pavetext.C
                 visualisation/graphics/tmathtext.C visualisation/graphics/tmathtext2.C
                 visualisation/graphs/gr106_exclusiongraph.C
                 visualisation/graphs/gr016_struct.C
                 hist/hist102_TH2_contour_list.C
                 hist/hist006_TH1_bar_charts.C
                 hist/hist037_TH2Poly_boxes.C
                 hist/hist060_TH1_stats.C
                 hist/hist014_TH1_cumulative.C
                 hist/hist004_TH1_labels.C
                 hist/hist036_TH2_labels.C
                 analysis/tree/h1analysis.C
                 math/chi2test.C
                 math/r/SimpleFitting.C)
#---Dependencies------------------------------------------------------
set(analysis-unfold-testUnfold5d-depends tutorial-analysis-unfold-testUnfold5c)
set(analysis-unfold-testUnfold5c-depends tutorial-analysis-unfold-testUnfold5b)
set(analysis-unfold-testUnfold5b-depends tutorial-analysis-unfold-testUnfold5a)
set(analysis-unfold-testUnfold7d-depends tutorial-analysis-unfold-testUnfold7c)
set(analysis-unfold-testUnfold7c-depends tutorial-analysis-unfold-testUnfold7b)
set(analysis-unfold-testUnfold7b-depends tutorial-analysis-unfold-testUnfold7a)
set(io-xml-xmlmodifyfile-depends tutorial-io-xml-xmlnewfile)
set(io-xml-xmlreadfile-depends tutorial-io-xml-xmlnewfile)
set(roofit-roofit-rf503_wspaceread-depends tutorial-roofit-roofit-rf502_wspacewrite)
set(io-readCode-depends tutorial-io-importCode)
set(math-fit-fit1-depends tutorial-hist-hist001_TH1_fillrandom)
set(math-fit-myfit-depends tutorial-math-fit-fitslicesy)
set(math-foam-foam_demopers-depends tutorial-math-foam-foam_demo)
set(io-tree-tree502_staff-depends  tutorial-io-tree-tree500_cernbuild)
set(io-tree-tree501_cernstaff-depends  tutorial-io-tree-tree500_cernbuild)
set(hist-hist006_TH1_bar_charts-depends  tutorial-io-tree-tree500_cernbuild)
set(benchmarks-depends tutorial-hsimple
                       tutorial-math-fit-fit1
                       tutorial-math-fit-myfit
                       tutorial-hist-hist015_TH1_read_and_draw
                       tutorial-hist-hist056_TPolyMarker_contour
                       tutorial-visualisation-geom-na49view
                       tutorial-io-tree-tree120_ntuple.C
                       tutorial-io-tree-spider
                       tutorial-io-hadd
                       tutorial-io-loopdir
                       tutorial-io-copyFiles)
set(visualisation-geom-na49view-depends tutorial-visualisation-geom-geometry)
set(io-tree-mt_readNtuplesFillHistosAndFit-depends tutorial-io-tree-mt_fillNtuples)
set(io-tree-mp_readNtuplesFillHistosAndFit-depends tutorial-io-tree-mp_fillNtuples)

#--many histfactory and roostats tutorials depending on having creating the file first with histfactory and example_combined_GaussExample_model.root
foreach(tname roofit-histfactory-hf001_example
              roofit-roostats-ModelInspector
              roofit-roostats-OneSidedFrequentistUpperLimitWithBands
              roofit-roostats-OneSidedFrequentistUpperLimitWithBands
              roofit-roostats-StandardBayesianMCMCDemo
              roofit-roostats-StandardBayesianNumericalDemo
              roofit-roostats-StandardFeldmanCousinsDemo
              roofit-roostats-StandardFrequentistDiscovery
              roofit-roostats-StandardHistFactoryPlotsWithCategories
              roofit-roostats-StandardHypoTestDemo
              roofit-roostats-StandardHypoTestInvDemo
              roofit-roostats-StandardProfileInspectorDemo
              roofit-roostats-StandardProfileLikelihoodDemo
              roofit-roostats-StandardTestStatDistributionDemo
              roofit-roostats-TwoSidedFrequentistUpperLimitWithBands)
  set(${tname}-depends tutorial-roofit-roostats-CreateExampleFile-py)
endforeach()

#--dependency for TMVA tutorials
set (machine_learning-TMVAClassificationApplication-depends tutorial-machine_learning-TMVAClassification)
set (machine_learning-TMVAClassificationCategory-depends tutorial-machine_learning-TMVAClassification)
set (machine_learning-TMVAClassificationCategoryApplication-depends tutorial-machine_learning-TMVAClassificationCategory)
set (machine_learning-TMVAMulticlass-depends tutorial-machine_learning-TMVAMultipleBackgroundExample)
set (machine_learning-TMVAMulticlassApplication-depends tutorial-machine_learning-TMVAMulticlass)
set (machine_learning-TMVARegressionApplication-depends tutorial-machine_learning-TMVARegression)
set (machine_learning-TMVACrossValidationRegression-depends tutorial-machine_learning-TMVARegressionApplication)
set (machine_learning-TMVACrossValidationApplication-depends tutorial-machine_learning-TMVACrossValidation)
set (machine_learning-tmva101_Training-depends tutorial-machine_learning-tmva100_DataPreparation-py)
set (machine_learning-tmva102_Testing-depends tutorial-machine_learning-tmva101_Training-py)
set (machine_learning-tmva003_RReader-depends tutorial-machine_learning-TMVAClassification)
set (machine_learning-tmva004_RStandardScaler-depends tutorial-machine_learning-tmva003_RReader)
if (ROOT_TORCH_FOUND)
  set (machine_learning-pytorch-ApplicationClassificationPyTorch-depends tutorial-machine_learning-pytorch-ClassificationPyTorch-py)
  set (machine_learning-pytorch-RegressionPyTorch-depends tutorial-machine_learning-pytorch-ApplicationClassificationPyTorch-py)
  set (machine_learning-pytorch-ApplicationRegressionPyTorch-depends tutorial-machine_learning-pytorch-RegressionPyTorch-py)
endif()
if (ROOT_KERAS_FOUND)
  set (machine_learning-TMVA_SOFIE_RSofieReader-depends tutorial-machine_learning-TMVA_Higgs_Classification)
  set (machine_learning-TMVA_SOFIE_RDataFrame_JIT-depends tutorial-machine_learning-TMVA_SOFIE_RSofieReader)
  set (machine_learning-TMVA_SOFIE_Keras_HiggsModel-depends tutorial-machine_learning-TMVA_SOFIE_RDataFrame_JIT)
  set (machine_learning-TMVA_SOFIE_RDataFrame-depends tutorial-machine_learning-TMVA_SOFIE_Keras_HiggsModel)
  set (machine_learning-TMVA_SOFIE_Inference-depends tutorial-machine_learning-TMVA_SOFIE_RDataFrame)
  if (ROOT_TORCH_FOUND)
    set (machine_learning-keras-RegressionKeras-depends tutorial-machine_learning-pytorch-RegressionPyTorch-py)
    set (machine_learning-keras-ClassificationKeras-depends tutorial-machine_learning-pytorch-ClassificationPyTorch-py)
  endif()
  set (machine_learning-keras-ApplicationRegressionKeras-depends tutorial-machine_learning-keras-RegressionKeras-py)
  set (machine_learning-keras-ApplicationClassificationKeras-depends tutorial-machine_learning-keras-ClassificationKeras-py)
endif()

#--List long-running tutorials to label them as "longtest"
set (long_running
     analysis/dataframe/df10[2-7]*
     analysis/mp_processSelector.C)
file(GLOB long_running RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${long_running})
#--List multithreaded tutorials to run them serially
set(NProcessors 4)
set (multithreaded_all_cores
     analysis/dataframe/df10[2-7]*
     experimental/rcanvas/df10*
    )
set (multithreaded
     ${multithreaded_all_cores}
     analysis/mp_processSelector.C
     machine_learning/TMVAMulticlass.C
     machine_learning/TMVA_CNN_Classification.C
     machine_learning/TMVA_Higgs_Classification.C
     machine_learning/TMVA_RNN_Classification.C
     machine_learning/TMVA_CNN_Classification.py
     machine_learning/TMVA_Higgs_Classification.py
     machine_learning/TMVA_RNN_Classification.py
     )
file(GLOB multithreaded_all_cores RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${multithreaded_all_cores})
file(GLOB multithreaded RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${multithreaded})

#---Loop over all tutorials and define the corresponding test---------

#---Define the primordial tutorials-----------------------------------
ROOT_ADD_TEST(tutorial-hsimple COMMAND ${ROOT_root_CMD} -b -l -n -q ${CMAKE_CURRENT_SOURCE_DIR}/hsimple.C
    PASSRC 255 FAILREGEX "Error in" "error:" "warning: Failed to call" LABELS tutorial ENVIRONMENT ${TUTORIAL_ENV})
if(ROOT_geom_FOUND)
  ROOT_ADD_TEST(tutorial-visualisation-geom-geometry COMMAND ${ROOT_root_CMD} -b -l -n -q ${CMAKE_CURRENT_SOURCE_DIR}/visualisation/geom/geometry.C
                FAILREGEX "Error in" "error:" "warning: Failed to call" LABELS tutorial ENVIRONMENT ${TUTORIAL_ENV})
endif()
# define Python GNN parsing tutorial needed to run before
if (ROOT_SONNET_FOUND AND ROOT_GRAPH_NETS_FOUND)
  ROOT_ADD_TEST(tutorial-machine_learning-TMVA_SOFIE_GNN_Parser COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/machine_learning/TMVA_SOFIE_GNN_Parser.py
  PASSRC 0 FAILREGEX "Error in" ": error:" LABELS tutorial ENVIRONMENT ${TUTORIAL_ENV})
  set (machine_learning-TMVA_SOFIE_GNN_Application-depends tutorial-machine_learning-TMVA_SOFIE_GNN_Parser)
endif()

#---Loop over all tutorials and define the corresponding test---------
foreach(t ${tutorials})
  list(FIND returncode_1 ${t} index)
  if(index EQUAL -1)
    set(rc 0)
  else()
    set(rc 255)
  endif()
  string(REPLACE ".C" "" tname ${t})
  string(REPLACE "/" "-" tname ${tname})

  set(labels tutorial)
  if(${t} IN_LIST long_running)
    list(APPEND labels longtest)
  endif()
  if(${t} IN_LIST multithreaded)
    list(APPEND labels multithreaded)
    # If this is not a TMVA tutorial, we want to limit the size of the thread
    # pool in case the tutorial invokes ROOT::EnableImplicitMT(), which by
    # default creates a thread pool of the size of the total number of cores.
    if(${t} IN_LIST multithreaded_all_cores)
      set(createThreadPool "-e \"ROOT::EnableImplicitMT(${NProcessors})\"")
    endif()
  else()
    unset(createThreadPool)
  endif()

  if(${t} IN_LIST need_network)
    list(APPEND labels needs_network)
  endif()

   # These tests on ARM64 need much more than 20 minutes - increase the timeout
   if(ROOT_ARCHITECTURE MATCHES arm64 OR ROOT_ARCHITECTURE MATCHES ppc64)
     set(thisTestTimeout 3000) # 50m
   else()
     set(thisTestTimeout 1200) # 20m
   endif()

  ROOT_ADD_TEST(tutorial-${tname}
                COMMAND ${ROOT_root_CMD} -b -l -q ${createThreadPool} ${CMAKE_CURRENT_SOURCE_DIR}/${t}${${tname}-aclic}
                PASSRC ${rc} FAILREGEX "Error in <" ": error:" "segmentation violation" "FROM HESSE     STATUS=FAILED" "warning: Failed to call"
                LABELS ${labels}
                DEPENDS tutorial-hsimple ${${tname}-depends}
                ENVIRONMENT ${TUTORIAL_ENV}
                TIMEOUT ${thisTestTimeout})

  if(${t} IN_LIST multithreaded)
    # Makes sure that this doesn't run in parallel with other multithreaded tutorials, and that cmake doesn't start too
    # many other tests. That we use 4 processors is actually a lie, because IMT takes whatever it finds.
    # However, even this poor indication of MT behaviour is a good hint for cmake to reduce congestion.
    set_tests_properties(tutorial-${tname} PROPERTIES RESOURCE_LOCK multithreaded PROCESSORS ${NProcessors})
  endif()
endforeach()

#---Loop over all MPI tutorials and define the corresponding test---------
foreach(t ${mpi_tutorials})
  list(FIND returncode_1 ${t} index)
  if(index EQUAL -1)
    set(rc 0)
  else()
    set(rc 255)
  endif()
  string(REPLACE ".C" "" tname ${t})
  string(REPLACE "/" "-" tname ${tname})

   # These tests on ARM64 need much more than 20 minutes - increase the timeout
   if(ROOT_ARCHITECTURE MATCHES arm64 OR ROOT_ARCHITECTURE MATCHES ppc64)
     set(thisTestTimeout 3000) # 50m
   else()
     set(thisTestTimeout 1200) # 20m
   endif()

  ROOT_ADD_TEST(tutorial-${tname}
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${ROOT_root_CMD} -b -l -q ${CMAKE_CURRENT_SOURCE_DIR}/${t}${${tname}-aclic}
                PASSRC ${rc} FAILREGEX "Error in <" ": error:" "segmentation violation" "FROM HESSE     STATUS=FAILED" "warning: Failed to call"
                LABELS tutorial
                DEPENDS tutorial-hsimple ${${tname}-depends}
                ENVIRONMENT ${TUTORIAL_ENV}
                TIMEOUT ${thisTestTimeout})
endforeach()

#---Python tutorials-----------------------------------------------------
if(ROOT_pyroot_FOUND)

  # Copy .rootlogon.py file into the build directory. It disables graphics for the Python tutorials
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/.rootlogon.py ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
  # Copy .enableImplicitMTWrapper.py file into the build directory. It can limit the size of the thread pool
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/.enableImplicitMTWrapper.py ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

  file(GLOB_RECURSE pytutorials RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py)

  # Now python-specific vetos:
  set(pyveto demos.py         # requires GUI
             math/fit/fit1_py.py      # not a tutorial
             visualisation/gui/gui_simple.py       # requires GUI
             io/tree/csv2tntuple.py          # not really a tutorial
             visualisation/geom/na49geomfile.py # ????
             visualisation/geom/na49visible.py  # ????
             io/tree/csv2tree_ReadStream.py # not a tutorial
             visualisation/gui/numberEntry.py  # requires GUI
             legacy/pyroot/*py      # legacy ...
             roofit/histfactory/makeQuickModel.py # not a tutorial
             visualisation/eve/lineset.py         # requires GUI
             io/sql/sqlcreatedb.py     # same as the C++ case
             io/sql/sqlfilldb.py       # same as the C++ case
             io/sql/sqlselect.py       # same as the C++ case
             launcher.py            # Not a tutorial
             .enableImplicitMTWrapper.py # Not a tutorial
             )

  if(NOT dataframe
     OR DEFINED ENV{ROOTTEST_IGNORE_NUMBA_PY3}
     OR (MSVC AND NOT win_broken_tests))
    list(APPEND pyveto analysis/dataframe/df038_NumbaDeclare.py)
  endif()

  if(dataframe AND DEFINED ENV{ROOTTEST_IGNORE_PANDAS_PY3})
    list(APPEND pyveto analysis/dataframe/df026_AsNumpyArrays.py)
  endif()

  # Rules specific to distributed RDataFrame
  # Disable distributed RDF tutorials if we didn't check dependencies in the environment first
  if(NOT test_distrdf_pyspark)
    list(APPEND pyveto analysis/dataframe/distrdf001_spark_connection.py)
  endif()
  if(NOT test_distrdf_dask)
    list(APPEND pyveto analysis/dataframe/distrdf002_dask_connection.py)
    list(APPEND pyveto analysis/dataframe/distrdf003_live_visualization.py)
  endif()
  # Use main Python executable to run in PySpark driver and executors
  if(test_distrdf_pyspark)
    list(APPEND TUTORIAL_ENV PYSPARK_PYTHON=${Python3_EXECUTABLE})
    if(MACOSX_VERSION VERSION_GREATER_EQUAL 10.13)
      # MacOS has changed rules about forking processes after 10.13
      # Running pyspark tests with XCode Python3 throws crashes with errors like:
      # `objc[17271]: +[__NSCFConstantString initialize] may have been in progress in another thread when fork() was called.`
      # This issue should have been fixed after Python 3.8 (see https://bugs.python.org/issue33725)
      # Indeed, any other Python 3.8+ executable does not show this crash. It is
      # specifically the XCode Python executable that triggers this.
      # For now, there seems no other way than this workaround,
      # which effectively sets the behaviour of `fork` back to MacOS 10.12
      list(APPEND TUTORIAL_ENV OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES)
    endif()
  endif()
  # These lists keep track of distrdf tutorials, so we can add specific properties later
  file(GLOB distrdf_spark_tutorials RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} analysis/dataframe/*spark*)
  file(GLOB distrdf_dask_tutorials RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} analysis/dataframe/*dask*)

  # Disable tutorial showing connection to the HTCondor service at CERN
  list(APPEND pyveto analysis/dataframe/distrdf004_dask_lxbatch.py)

  ROOT_FIND_PYTHON_MODULE(xgboost QUIET)
  if(NOT ROOT_XGBOOST_FOUND OR NOT dataframe)
    file(GLOB tmva_veto_py RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} machine_learning/tmva10*.py roofit/roofit/rf618_mixture_models.py)
    list(APPEND pyveto ${tmva_veto_py})
  endif()

  if(NOT ROOT_KERAS_FOUND)
    file(GLOB tmva_veto_py RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} machine_learning/keras/*.py)
    list(APPEND pyveto ${tmva_veto_py})
  elseif(ROOT_ARCHITECTURE MATCHES macosx)
    #veto also keras tutorial on macos due to issue in disabling eager execution on macos
    list(APPEND pyveto machine_learning/keras/RegressionKeras.py)
    list(APPEND pyveto machine_learning/keras/ApplicationRegressionKeras.py)
    list(APPEND pyveto machine_learning/keras/MultiClassKeras.py)
  endif()

  if (ROOT_KERAS_FOUND)
    set (machine_learning-TMVA_SOFIE_RDataFrame-py-depends tutorial-machine_learning-TMVA_SOFIE_Keras_HiggsModel)
  endif()

  if(NOT ROOT_TORCH_FOUND)
    file(GLOB tmva_veto_py RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} machine_learning/pytorch/*.py)
    list(APPEND pyveto ${tmva_veto_py})
  endif()
  # disable PyTorch model file used by TMVA_CNN_Classification.C
  list(APPEND pyveto machine_learning/PyTorch_Generate_CNN_Model.py)

  if(NOT ROOT_geom_FOUND)
    list(APPEND pyveto geom/geometry.py)
  endif()

  # Now glob all vetos for pyroot
  file(GLOB pyveto RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${pyveto})

  list(LENGTH pytutorials nTotal)
  # Apply global .C/.py veto from above:
  list(REMOVE_ITEM pytutorials ${tutorials_veto})
  list(REMOVE_ITEM pytutorials ${pyveto})
  list(LENGTH pytutorials nAfterVeto)

  message(STATUS "${nAfterVeto}/${nTotal} python tutorials have been activated.")

  #---Python tutorials dependencies--------------------------------------
  set(io-tree-ntuple1-depends tutorial-hsimple-py)
  set(hist-hist015_TH1_read_and_draw-depends tutorial-hsimple-py)
  set(pyroot-benchmarks-depends tutorial-hsimple-py
                                tutorial-math-fit-fit1-py
                                tutorial-visualisation-geom-na49view-py
                                tutorial-hist-hist015_TH1_read_and_draw-py
                                tutorial-io-tree-ntuple1-py)
  set(math-fit-fit1-depends tutorial-hist-hist001_TH1_fillrandom-py)
  set(visualisation-geom-na49view-depends tutorial-visualisation-geom-geometry-py)
  set(roofit-roofit-rf503_wspaceread-depends tutorial-roofit-roofit-rf502_wspacewrite-py)
  set(roofit-roofit-rf618_mixture_models-depends tutorial-analysis-dataframe-df106_HiggsToFourLeptons-py)

  # Avoid a race condition: make sure Python tutorial is run after C++ tutorial
  set(roofit-roofit-rf104_classfactory-depends tutorial-roofit-roofit-rf104_classfactory)
  set(roofit-roofit-rf512_wsfactory_oper-depends tutorial-roofit-roofit-rf512_wsfactory_oper)
  set(machine_learning-TMVA_Higgs_Classification-depends tutorial-machine_learning-TMVA_Higgs_Classification)
  set(machine_learning-TMVA_CNN_Classification-depends tutorial-machine_learning-TMVA_CNN_Classification)
  set(machine_learning-TMVA_RNN_Classification-depends tutorial-machine_learning-TMVA_RNN_Classification)
  set(roofit-roostats-CreateExampleFile-depends tutorial-roofit-roostats-CreateExampleFile)

  #----------------------------------------------------------------------
  # List requirements for python tutorials.
  # To add a new requirement, add a glob expression that's named requires_<packageName>,
  # and add it to the list "fixtureLists" below.
  file(GLOB requires_numpy   RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
      analysis/dataframe/df026_AsNumpyArrays.py
      analysis/dataframe/df032_RDFFromNumpy.py
      math/fit/combinedFit.py
      math/fit/multifit.py
      math/fit/NumericalMinimization.py
      roofit/roofit/rf409_NumPyPandasToRooFit.py)
  file(GLOB requires_numba   RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} analysis/dataframe/df038_NumbaDeclare.py)
  file(GLOB requires_pandas  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
      analysis/dataframe/df026_AsNumpyArrays.py
      analysis/dataframe/df035_RDFFromPandas.py
      roofit/roofit/rf409_NumPyPandasToRooFit.py)
  file(GLOB requires_keras   RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} machine_learning/keras/*.py)
  file(GLOB requires_torch   RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} machine_learning/pytorch/*.py)
  file(GLOB requires_xgboost RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} machine_learning/tmva10*.py)
  set(fixtureLists requires_numpy requires_numba requires_pandas requires_keras requires_xgboost requires_torch)

  # Now set up all the tests
  foreach(t ${pytutorials})
    if (${t} IN_LIST returncode_1)
      set(rc 255)
    else()
      set(rc 0)
    endif()

    set(labels tutorial)
    if(${t} IN_LIST long_running)
      list(APPEND labels longtest)
    endif()
    if(${t} IN_LIST multithreaded)
      list(APPEND labels multithreaded)
      # If this is not a TMVA tutorial, we want to limit the size of the thread
      # pool in case the tutorial invokes ROOT::EnableImplicitMT(), which by
      # default creates a thread pool of the size of the total number of cores.
      if(${t} IN_LIST multithreaded_all_cores)
        set(setThreadPoolSize ".enableImplicitMTWrapper.py")
        set(thisTestPoolSize ${NProcessors})
      endif()
    else()
      unset(setThreadPoolSize)
      unset(thisTestPoolSize)
    endif()

    string(REPLACE ".py" "" tname ${t})
    string(REPLACE "/" "-" tname ${tname})

    set(tutorial_name tutorial-${tname}-py)

    list(FIND pyexp_fail ${tutorial_name} index)
    if(index EQUAL -1)
      set(py_will_fail "")
    else()
      set(py_will_fail ${PYTESTS_WILLFAIL})
    endif()

    # Test if this tutorial is requiring any fixture
    unset(python_deps)
    foreach(fixtureList ${fixtureLists})
      if(${t} IN_LIST ${fixtureList})
        string(REPLACE "requires_" "" fixture ${fixtureList})
        list(APPEND python_deps ${fixture})
        list(APPEND labels python_runtime_deps)
      endif()
    endforeach()

    ROOT_ADD_TEST(${tutorial_name}
                COMMAND ${Python3_EXECUTABLE} ${setThreadPoolSize} ${thisTestPoolSize} ${CMAKE_CURRENT_SOURCE_DIR}/${t}
                PASSRC ${rc} FAILREGEX "Error in" ": error:" "segmentation violation"
                LABELS ${labels}
                DEPENDS ${${tname}-depends}
                ENVIRONMENT ${TUTORIAL_ENV}
                PYTHON_DEPS ${python_deps}
                ${py_will_fail})

    if(${t} IN_LIST multithreaded)
      # Makes sure that this doesn't run in parallel with other multithreaded tutorials, and that cmake doesn't start too
      # many other tests. That we use 4 processors is actually a lie, because IMT takes whatever it finds.
      # However, even this poor indication of MT behaviour is a good hint for cmake to reduce congestion.
      set_tests_properties(${tutorial_name} PROPERTIES RESOURCE_LOCK multithreaded PROCESSORS ${NProcessors})
    endif()

    if(${t} IN_LIST distrdf_spark_tutorials)
      # Create a resource lock for the creation of a Spark cluster. This is also used in roottest.
      # Also signal 4 processors to cmake to give the tutorial some room (it uses 2 cores).
      set_tests_properties(${tutorial_name} PROPERTIES RESOURCE_LOCK spark_resource_lock PROCESSORS ${NProcessors})
    endif()

    if(${t} IN_LIST distrdf_dask_tutorials)
      # Create a resource lock for the creation of a Dask cluster. This is also used in roottest.
      # Also signal 4 processors to cmake to give the tutorial some room (it uses 2 cores).
      set_tests_properties(${tutorial_name} PROPERTIES RESOURCE_LOCK dask_resource_lock PROCESSORS ${NProcessors})
    endif()

  endforeach()
endif()
