# CMakeLists.txt for the ROOT tutorials programs.
# Author: Pere Mato, 25/10/2010
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(tutorials)
find_package(ROOT REQUIRED)

if(DEFINED ROOT_SOURCE_DIR)  # Testing using the binary tree
  set(ROOT_root_CMD root.exe)
  if(NOT WIN32)  # Ignore environment on Windows for the time being
    set(ROOT_environ PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH}
                     ${ld_library_path}=${CMAKE_BINARY_DIR}/lib:$ENV{${ld_library_path}}
                     ROOTSYS=${CMAKE_BINARY_DIR}
                     PYTHONPATH=${CMAKE_BINARY_DIR}/lib:$ENV{PYTHONPATH})
  else()
    set(ROOT_environ PYTHONPATH=${CMAKE_BINARY_DIR}/bin;$ENV{PYTHONPATH})
  endif()
else()                       # testing using an installation
  if(DEFINED ROOT_USE_FILE)
    include(${ROOT_USE_FILE})
  else()  #---If ROOT was built with the classic configure/make---
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules)
    include(RootNewMacros)
    set(ROOT_root_CMD root.exe)
  endif()
  enable_testing()
endif()

#---Copy the CTestCustom.cmake file into the build directory--------
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

#---Provide a rootlogon.C file in the current build directory that
#   will affect the way we run all tutorials.
#   This overwrites the existing rootlogon.C and rootalias.C in the
#   tutorials directory which is copied to the build area.
#-------------------------------------------------------------------
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/rootlogon.C "{
  // Needed by ACLiC to use the current dicrectory for scratch area
  gSystem->SetBuildDir(\".\", kTRUE);
}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/rootalias.C "")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/.rootrc "
Proof.Sandbox: /tmp/proof
")

#---Tutorials disabled depending on the build components-------------
if(NOT ROOT_minuit2_FOUND)
  set(minuit2_veto fit/fit2dHist.C fit/fit2dHist.C
                   fit/fitCircle.C fit/minuit2FitBench2D.C
                   fit/minuit2FitBench2D.C fit/minuit2FitBench.C
                   fit/minuit2FitBench.C fit/minuit2GausFit.C
                   fit/minuit2GausFit.C fit/NumericalMinimization.C
                   fit/combinedFit.C fit/TestBinomial.C)
endif()

if(NOT ROOT_roofit_FOUND)
  set(roofit_veto  fit/RoofitDemo.C
                   roofit/*.C roostats/*.C histfactory/*.C)
endif()

if(NOT ROOT_unuran_FOUND)
  set(unuran_veto  math/testrandom.C unuran/unuranDemo.C unuran/unuranFoamTest.C)
endif()

if(NOT ROOT_xml_FOUND)
  set(xml_veto  xml/*.C
                histfactory/*.C)   # histfactory requires xml
endif()

if(WIN32)
  set(histfactory_veto histfactory/*.C
                       roostats/StandardFrequentistDiscovery.C) # histfactory doesn't work on Windows
endif()

if(NOT ROOT_fitsio_FOUND)
  set(fitsio_veto  fitsio/*.C)
endif()

if(NOT ROOT_mathmore_FOUND)
  set(mathmore_veto
      math/quasirandom.C
      math/exampleMultiRoot.C
      math/Bessel.C
      math/LegendreAssoc.C
      math/Legendre.C
      math/mathmoreIntegration.C
      math/tStudent.C
      math/normalDist.C)
endif()

if(NOT ROOT_fftw3_FOUND)
  set(fftw3_veto roofit/rf208_convolution.C
                 roofit/rf210_angularconv.C
                 roofit/rf211_paramconv.C
                 roofit/rf512_wsfactory_oper.C
                 fft/FFT.C)
endif()

if(NOT ROOT_opengl_FOUND)
  set(opengl_veto tree/staff.C)
endif()

if(NOT GRAPHVIZ_FOUND)
  set(gviz_veto graphs/graphstruct.C)
endif()

#---These ones requires a display to run-----------------------------
set(gui_veto fit/fitpanel_playback.C
             cocoa/*.C
             geom/*.C
             gl/*.C
             gui/*.C
             hist/exec1.C
             hist/exec2.C
             image/*.C
             graphics/psview.C graphics/gtime.C
             graphics/graph_edit_playback.C
             roostats/ModelInspector.C
             tree/tvdemo.C
             eve/*.C)

#---These ones are disabled !!! ------------------------------------
set(extra_veto
  htmlex.C
  rootalias.C          # Helper macro
  rootlogon.C          # Helper macro
  rootlogoff.C         # Helper macro
  rootmarks.C          # Instrumentation. Not a standalone tutorial
  memstat/memstat.C    # Instrumentation. Not a standalone tutorial
  html/*.C
  net/*.C
  proof/*.C
  pythia/*.C
  sql/*.C
  tree/hsimpleProxy.C  # A driver uses this macro which cannot be executed directly
  tree/tree0.C
  tree/tree2a.C
  tree/tree4.C
  roostats/rs401d_FeldmanCousins.C  # Takes too much time
  roofit/rf104_classfactory.C
  histfactory/ModifyInterpolation.C
  tree/copytree2.C
  tree/copytree3.C
  tree/copytree.C
  http/*.C)

set(all_veto hsimple.C
             ${extra_veto}
             ${gui_veto}
             ${minuit2_veto}
             ${roofit_veto}
             ${unuran_veto}
             ${xml_veto}
             ${fitsio_veto}
             ${mathmore_veto}
             ${fftw3_veto}
             ${opengl_veto}
             ${gviz_veto}
             ${histfactory_veto})

file(GLOB_RECURSE tutorials RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.C)
file(GLOB tutorials_veto RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${all_veto})

list(LENGTH tutorials total)
list(LENGTH tutorials_veto veto)
message(STATUS "${veto}/${total} tutorials have been vetoed for various reasons")

list(REMOVE_ITEM tutorials ${tutorials_veto})

#---Special return code------------------------------------------------
set(returncode_1 fit/fit2a.C fit/graph2dfit.C
                 graphics/archi.C graphics/arrow.C
                 graphics/crown.C graphics/diamond.C
                 graphics/earth.C graphics/ellipse.C
                 graphics/event.C graphics/pavetext.C
                 graphics/tmathtext.C graphics/tmathtext2.C
                 graphs/exclusiongraph.C graphs/multigraph.C
                 graphs/multipalette.C
                 graphs/graphstruct.C
                 hist/ContourList.C hist/hstack.C
                 hist/h2proj.C
                 hist/th2polyBoxes.C
                 hist/statsEditing.C
                 hist/cumulative.C
                 tree/h1analysis.C
                 io/fildir.C
                 math/chi2test.C )
#---Dependencies------------------------------------------------------
set(math-testUnfold5d-depends tutorial-math-testUnfold5c)
set(math-testUnfold5c-depends tutorial-math-testUnfold5b)
set(math-testUnfold5b-depends tutorial-math-testUnfold5a)
set(xml-xmlreadfile-depends tutorial-xml-xmlnewfile)
set(roofit-rf503_wspaceread-depends tutorial-roofit-rf502_wspacewrite)
set(histfactory-example-depends tutorial-histfactory-makeExample)
set(io-readCode-depends tutorial-io-importCode)
set(fit-fit1-depends tutorial-hist-fillrandom)
set(foam-foam_demopers-depends tutorial-foam-foam_demo)
set(tree-staff-depends  tutorial-tree-cernbuild)
set(tree-cernstaff-depends  tutorial-tree-cernbuild)
set(hist-hbars-depends  tutorial-tree-cernbuild)
set(benchmarks-depends tutorial-hsimple
                       tutorial-fit-fit1
                       tutorial-fit-myfit
                       tutorial-fit-fitslicesy
                       tutorial-hist-h1draw
                       tutorial-geom-na49view
                       tutorial-tree-ntuple1
                       tutorial-io-hadd
                       tutorial-io-copyFiles)

#--many roostats tutorials depending on having creating the file first with histfactory
foreach(tname  ModelInspector OneSidedFrequentistUpperLimitWithBands StandardBayesianMCMCDemo StandardBayesianNumericalDemo
               StandardFeldmanCousinsDemo  StandardFrequentistDiscovery StandardHistFactoryPlotsWithCategories StandardHypoTestDemo
               StandardHypoTestInvDemo StandardProfileInspectorDemo StandardTestStatDistributionDemo
               OneSidedFrequentistUpperLimitWithBands TwoSidedFrequentistUpperLimitWithBands)
  set(roostats-${tname}-depends tutorial-roostats-StandardProfileLikelihoodDemo)
endforeach()

#--With ACLiC---------------------------------------------------------
set(fit-combinedFit-aclic ++)
set(fit-line3Dfit-aclic ++)
set(fit-TwoHistoFit2D-aclic ++)
set(math-testUnfold4-aclic ++)
set(io-double32-aclic ++)
set(tree-hvector-aclic ++)
set(math-testrandom-aclic ++)

#---Define the primordial tutorials-----------------------------------
ROOT_ADD_TEST(tutorial-hsimple COMMAND ${ROOT_root_CMD} -b -l -n -q ${CMAKE_CURRENT_SOURCE_DIR}/hsimple.C
                PASSRC 1 FAILREGEX "Error in" "error:" LABELS tutorial)
ROOT_ADD_TEST(tutorial-geom-geometry COMMAND ${ROOT_root_CMD} -b -l -n -q ${CMAKE_CURRENT_SOURCE_DIR}/geom/geometry.C
                FAILREGEX "Error in" "error:" LABELS tutorial)

#---Loop over all tutorials and define the corresponding test---------
foreach(t ${tutorials})
  list(FIND returncode_1 ${t} index)
  if(index EQUAL -1)
    set(rc 0)
  else()
    set(rc 1)
  endif()
  string(REPLACE ".C" "" tname ${t})
  string(REPLACE "/" "-" tname ${tname})
  ROOT_ADD_TEST(tutorial-${tname}
                COMMAND ${ROOT_root_CMD} -b -l -q ${CMAKE_CURRENT_SOURCE_DIR}/${t}${${tname}-aclic}
                PASSRC ${rc} FAILREGEX "Error in" "error:" "segmentation violation"
                LABELS tutorial
                DEPENDS tutorial-hsimple ${${tname}-depends}
                ENVIRONMENT ${ROOT_environ})
endforeach()


#---Python tutorials-----------------------------------------------------
if(ROOT_python_FOUND)
  find_package(PythonInterp REQUIRED)

  file(GLOB pytutorials RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} pyroot/*.py)
  set(pyveto pyroot/demo.py         # requires GUI
             pyroot/fit1_py.py      # not a tutorial
             pyroot/gui_ex.py       # requires GUI
             pyroot/mrt.py          # not really a tutorial
             pyroot/na49geomfile.py # ????
             pyroot/na49visible.py  # ????
             pyroot/parse_CSV_file_with_TTree_ReadStream.py # not a tutorial
             pyroot/qtexample.py    # requires GUI/Qt
             pyroot/numberEntry.py  # requires GUI
             )
  list(REMOVE_ITEM pytutorials ${pyveto})

  #---Python tutorials dependencies--------------------------------------
  set(pyroot-ntuple1-depends tutorial-pyroot-hsimple)
  set(pyroot-h1draw-depends tutorial-pyroot-hsimple)
  set(pyroot-benchmarks-depends tutorial-pyroot-hsimple
                                tutorial-pyroot-fit1
                                tutorial-pyroot-na49view
                                tutorial-pyroot-h1draw
                                tutorial-pyroot-ntuple1)
  set(pyroot-fit1-depends tutorial-pyroot-fillrandom)
  set(pyroot-na49view-depends tutorial-pyroot-geometry)

  foreach(t ${pytutorials})
    string(REPLACE ".py" "" tname ${t})
    string(REPLACE "/" "-" tname ${tname})
    ROOT_ADD_TEST(tutorial-${tname}
                COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${t} -b
                FAILREGEX "Error in" "error:" "segmentation violation"
                LABELS tutorial
                DEPENDS ${${tname}-depends}
                ENVIRONMENT ${ROOT_environ})
  endforeach()
endif()









