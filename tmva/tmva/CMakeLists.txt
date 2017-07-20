############################################################################
# CMakeLists.txt file for building ROOT tmva package
# @author Pere Mato, CERN
############################################################################
#---The following is needed because client codes of TMVA (in ROOT itself) assumes to find header files under /TMVA
#execute_process(COMMAND cmake -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/inc/TMVA)

if(NOT CMAKE_PROJECT_NAME STREQUAL ROOT)
  cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
  project(TMVA)
  include(${ROOTSYS}/cmake/modules/StandaloneBuild.cmake)
endif()

set(headers1 Configurable.h Factory.h  MethodBase.h MethodCompositeBase.h
	     MethodANNBase.h MethodTMlpANN.h MethodRuleFit.h MethodCuts.h MethodFisher.h
	     MethodKNN.h MethodCFMlpANN.h MethodCFMlpANN_Utils.h MethodLikelihood.h
	     MethodHMatrix.h MethodPDERS.h MethodBDT.h MethodDT.h MethodSVM.h MethodBayesClassifier.h
	     MethodFDA.h MethodMLP.h MethodBoost.h
	     MethodPDEFoam.h MethodLD.h MethodCategory.h MethodDNN.h)
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
set(headers4 NeuralNet.h TNeuron.h TSynapse.h TActivationChooser.h TActivation.h TActivationSigmoid.h TActivationIdentity.h
         TActivationTanh.h TActivationRadial.h TActivationReLU.h TNeuronInputChooser.h TNeuronInput.h TNeuronInputSum.h
	     TNeuronInputSqSum.h TNeuronInputAbs.h Types.h Ranking.h RuleFit.h RuleFitAPI.h IMethod.h MsgLogger.h
	     VariableTransformBase.h VarTransformHandler.h VariableIdentityTransform.h VariableDecorrTransform.h VariablePCATransform.h
	     VariableGaussTransform.h VariableNormalizeTransform.h VariableRearrangeTransform.h VariableTransform.h ROCCalc.h ROCCurve.h)

set(headers5 Envelope.h VariableImportance.h CrossValidation.h HyperParameterOptimisation.h Event.h Results.h ResultsClassification.h ResultsRegression.h ResultsMulticlass.h VariableInfo.h ClassInfo.h DataLoader.h DataSet.h DataSetInfo.h DataInputHandler.h DataSetManager.h DataSetFactory.h LossFunction.h)

#---Need to suffix each header name by TMVA/  -----------------
foreach(hs headers1 headers2 headers3 headers4 headers5)
  foreach(h ${${hs}})
    set(t${hs} ${t${hs}} TMVA/${h})
  endforeach()
endforeach()

#---Assign source files to the implementations -----------------
SET(DNN_FILES      src/DNN/Architectures/Reference.cxx
                   src/DNN/Architectures/Reference/DataLoader.cxx)
SET(DNN_CUDA_FILES src/DNN/Architectures/Cuda.cu
                   src/DNN/Architectures/Cuda/CudaBuffers.cxx
                   src/DNN/Architectures/Cuda/CudaMatrix.cu)
SET(DNN_CPU_FILES  src/DNN/Architectures/Cpu.cxx
                   src/DNN/Architectures/Cpu/CpuBuffer.cxx
                   src/DNN/Architectures/Cpu/CpuMatrix.cxx)

#---Handle CUDA dependent code. -----------------
if(CUDA_FOUND)
  CUDA_ADD_LIBRARY(dnn_cuda ${DNN_CUDA_FILES})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDNNCUDA")
  set(DNN_CUDA_LIBRARIES dnn_cuda ${CUDA_CUBLAS_LIBRARIES})
  add_dependencies(dnn_cuda move_headers)
else()
  set(DNN_CUDA_LIBRARIES)
  set(installoptions ${installoptions} FILTER "Cuda")
endif()

#---Handle BLAS dependent code. -----------------
if(BLAS_FOUND AND imt)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDNNCPU")
  set(DNN_CPU_LIBRARIES MathCore Matrix ${BLAS_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
  include_directories(SYSTEM ${TBB_INCLUDE_DIRS})
else()
  set(DNN_CPU_LIBRARIES)
  set(DNN_CPU_FILES)
  set(installoptions ${installoptions} FILTER "Cpu")
endif()

ROOT_GENERATE_DICTIONARY(G__TMVA ${theaders1} ${theaders2} ${theaders3} ${theaders4} ${theaders5}  MODULE TMVA LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM" DEPENDENCIES RIO Hist Tree TreePlayer MLP Minuit XMLIO)

ROOT_LINKER_LIBRARY(TMVA *.cxx G__TMVA.cxx ${DNN_FILES} ${DNN_CPU_FILES}
                    LIBRARIES Core ${DNN_CUDA_LIBRARIES} ${DNN_CPU_LIBRARIES}
                    DEPENDENCIES RIO Hist Tree TreePlayer MLP Minuit XMLIO)

ROOT_INSTALL_HEADERS(${installoptions})

if(NOT gnuinstall)
  install(DIRECTORY test DESTINATION tmva 
  FILES_MATCHING PATTERN "*.C" 
                 PATTERN "*.gif" 
                 PATTERN "*.png" 
                 PATTERN "README"
                 PATTERN "data" EXCLUDE)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test/DNN)
ROOT_ADD_TEST_SUBDIRECTORY(test/ROC)




