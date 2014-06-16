# CMakeLists.txt for the ROOT tutorials programs.
# Author: Pere Mato, 25/10/2010
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(tutorials)
find_package(ROOT REQUIRED)

if(DEFINED ROOT_SOURCE_DIR)  # Testing using the binary tree
  set(ROOT_root_CMD root.exe)
endif()

#---Definition of the helper function--------------------------------
function(ROOT_ADD_TUTORIAL macrofile rc)
  string(REPLACE ".C" "" name ${macrofile})
  string(REPLACE "/" "-" name ${name})
  ROOT_ADD_TEST(tutorial-${name} COMMAND ${ROOT_root_CMD} -b -l -n -q ${CMAKE_CURRENT_SOURCE_DIR}/${macrofile}
                PASSRC ${rc} FAILREGEX "Error in" "error:" LABELS tutorial)
endfunction()

#---Tutorials disabled depending on the build components-------------
if(NOT ROOT_minuit2_FOUND)
  set(minuit2_veto fit/fit2dHist.C fit/fit2dHist.C
                   fit/fitCircle.C fit/minuit2FitBench2D.C
                   fit/minuit2FitBench2D.C fit/minuit2FitBench.C
                   fit/minuit2FitBench.C fit/minuit2GausFit.C
                   fit/minuit2GausFit.C fit/NumericalMinimization.C
                   fit/combinedFit.C)
endif()

if(NOT ROOT_roofit_FOUND)
  set(roofit_veto  fit/RoofitDemo.C 
                   roofit/*.C roostats/*.C histfactory/*.C)
endif()

if(NOT ROOT_unuran_FOUND)
  set(unuran_veto  math/testrandom.C unuran/unuranDemo.C unuran/unuranFoamTest.C)
endif()

if(NOT ROOT_xml_FOUND)
  set(xml_veto  xml/*.C)
endif()

if(NOT ROOT_fitsio_FOUND)
  set(fitsio_veto  fitsio/*.C)
endif()

if(NOT ROOT_mathmore_FOUND)
  set(mathmore_veto math/quasirandom.C math/exampleMultiRoot.C)
endif()

if(NOT ROOT_fftw3_FOUND)
  set(fftw3_veto roofit/rf208_convolution.C
                 roofit/rf210_angularconv.C
                 roofit/rf211_paramconv.C
                 roofit/rf512_wsfactory_oper.C)
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
             hist/exec1.C hist/exec2.C
             image/*.C
             graphics/psview.C graphics/gtime.C
             graphics/graph_edit_playback.C
             tree/tvdemo.C
             eve/*.C)

#---These ones are disabled !!! ------------------------------------
set(extra_veto
  htmlex.C
  rootalias.C
  rootlogon.C
  rootlogoff.C
  fft/FFT.C
  fit/fit1_C.C  fit/TwoHistoFit2D.C fit/line3Dfit.C fit/MDF.C
  foam/*.C
#  graphs/graphstruct.C
  html/*.C
  math/Bessel.C math/LegendreAssoc.C math/Legendre.C
  math/mathmoreIntegration.C math/normalDist.C math/tStudent.C
  math/testUnfold2.C math/pca.C
  net/*.C
  proof/*.C
  pythia/*.C
  sql/*.C
  tree/hsimpleProxy.C # A driver uses this macro which cannot be executed directly
  tree/tree0.C tree/tree2a.C tree/tree4.C
  roostats/rs401d_FeldmanCousins.C
  roofit/rf104_classfactory.C
  histfactory/ModifyInterpolation.C
  tree/copytree2.C tree/copytree3.C tree/copytree.C
  math/testUnfold4.C
  fit/minuit2FitBench.C
  fit/minuit2FitBench2D.C
  roostats/*.C
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
             ${gviz_veto})
             
file(GLOB_RECURSE tutorials RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.C)
file(GLOB tutorials_veto RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${all_veto})

list(LENGTH tutorials total)
list(LENGTH tutorials_veto veto)
message(STATUS "${veto}/${total} tutorials have been vetoed for various reasons")

list(REMOVE_ITEM tutorials ${tutorials_veto})

#---Special return code------------------------------------------------
set(returncode_1 fit/fit2a.C fit/graph2dfit.C fit/multidimfit.C
                 graphics/archi.C graphics/arrow.C
                 graphics/crown.C graphics/diamond.C
                 graphics/earth.C graphics/ellipse.C
                 graphics/event.C graphics/pavetext.C
                 graphics/tmathtext.C graphics/tmathtext2.C
                 graphs/exclusiongraph.C graphs/multigraph.C
                 graphs/multipalette.C 
                 hist/ContourList.C hist/hstack.C 
                 hist/th2polyBoxes.C
                 io/fildir.C 
                 math/chi2test.C )
#---Dependencies------------------------------------------------------
set(math-testUnfold5d-depends tutorial-math-testUnfold5c)
set(math-testUnfold5c-depends tutorial-math-testUnfold5b)
set(math-testUnfold5b-depends tutorial-math-testUnfold5a)
set(xml-xmlreadfile-depends tutorial-xml-xmlnewfile)
set(roofit-rf503_wspaceread-depends tutorial-roofit-rf502_wspacewrite)

#---Loop over all tutorials and define the corresponding test---------

ROOT_ADD_TEST(tutorial-hsimple COMMAND ${ROOT_root_CMD} -b -l -n -q ${CMAKE_CURRENT_SOURCE_DIR}/hsimple.C
                PASSRC 1 FAILREGEX "Error in" "error:" LABELS tutorial)
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
                COMMAND ${ROOT_root_CMD} -b -l -n -q ${CMAKE_CURRENT_SOURCE_DIR}/${t}
                PASSRC ${rc} FAILREGEX "Error in" "error:" "segmentation violation"
                LABELS tutorial
                DEPENDS tutorial-hsimple ${${name}-depends})
endforeach()


                   





