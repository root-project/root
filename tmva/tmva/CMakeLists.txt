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

ROOT_USE_PACKAGE(math/mathcore)
ROOT_USE_PACKAGE(math/minuit)
ROOT_USE_PACKAGE(math/mlp)
ROOT_USE_PACKAGE(hist/histpainter)
ROOT_USE_PACKAGE(tree/treeplayer)
ROOT_USE_PACKAGE(io/xml) 
#ROOT_USE_PACKAGE(gui/gui) 

set(headers1 Configurable.h Event.h Factory.h MethodBase.h MethodCompositeBase.h
	     MethodANNBase.h MethodTMlpANN.h MethodRuleFit.h MethodCuts.h MethodFisher.h
	     MethodKNN.h MethodCFMlpANN.h MethodCFMlpANN_Utils.h MethodLikelihood.h
	     MethodHMatrix.h MethodPDERS.h MethodBDT.h MethodDT.h MethodSVM.h MethodBayesClassifier.h
	     MethodFDA.h MethodMLP.h MethodBoost.h
	     MethodPDEFoam.h MethodLD.h MethodCategory.h)
set(headers2 TSpline2.h TSpline1.h PDF.h BinaryTree.h BinarySearchTreeNode.h BinarySearchTree.h 
	     Timer.h RootFinder.h CrossEntropy.h DecisionTree.h DecisionTreeNode.h MisClassificationError.h 
	     Node.h SdivSqrtSplusB.h SeparationBase.h RegressionVariance.h Tools.h Reader.h 
	     GeneticAlgorithm.h GeneticGenes.h GeneticPopulation.h GeneticRange.h GiniIndex.h 
	     GiniIndexWithLaplace.h SimulatedAnnealing.h QuickMVAProbEstimator.h)
set(headers3 Config.h KDEKernel.h Interval.h LogInterval.h FitterBase.h MCFitter.h GeneticFitter.h 
         SimulatedAnnealingFitter.h MinuitFitter.h MinuitWrapper.h IFitterTarget.h
         PDEFoam.h PDEFoamDecisionTree.h PDEFoamDensityBase.h PDEFoamDiscriminantDensity.h
         PDEFoamEventDensity.h PDEFoamTargetDensity.h PDEFoamDecisionTreeDensity.h PDEFoamMultiTarget.h
         PDEFoamVect.h PDEFoamCell.h PDEFoamDiscriminant.h PDEFoamEvent.h PDEFoamTarget.h
         PDEFoamKernelBase.h PDEFoamKernelTrivial.h PDEFoamKernelLinN.h PDEFoamKernelGauss.h
         BDTEventWrapper.h CCTreeWrapper.h
         CCPruner.h CostComplexityPruneTool.h SVEvent.h OptimizeConfigParameters.h)
set(headers4 TNeuron.h TSynapse.h TActivationChooser.h TActivation.h TActivationSigmoid.h TActivationIdentity.h
         TActivationTanh.h TActivationRadial.h TActivationReLU.h TNeuronInputChooser.h TNeuronInput.h TNeuronInputSum.h
	     TNeuronInputSqSum.h TNeuronInputAbs.h Types.h Ranking.h RuleFit.h RuleFitAPI.h IMethod.h MsgLogger.h
	     VariableTransformBase.h VariableIdentityTransform.h VariableDecorrTransform.h VariablePCATransform.h 
	     VariableGaussTransform.h VariableNormalizeTransform.h VariableRearrangeTransform.h ROCCalc.h)

#---Need to suffix each header name by TMVA/  -----------------
foreach(hs headers1 headers2 headers3 headers4)
  foreach(h ${${hs}})
    set(t${hs} ${t${hs}} TMVA/${h})
  endforeach()
endforeach()

ROOT_GENERATE_DICTIONARY(G__TMVA ${theaders1} ${theaders2} ${theaders3} ${theaders4}   MODULE TMVA LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")

ROOT_LINKER_LIBRARY(TMVA *.cxx G__TMVA.cxx LIBRARIES Core
                    DEPENDENCIES RIO Hist Tree TreeViewer  MLP Minuit XMLIO)

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




