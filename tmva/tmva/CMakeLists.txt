############################################################################
# CMakeLists.txt file for building ROOT tmva package
# @author Pere Mato, CERN
############################################################################

set(TMVA_HEADERS
  TMVA/BDTEventWrapper.h
  TMVA/BinarySearchTree.h
  TMVA/BinarySearchTreeNode.h
  TMVA/BinaryTree.h
  TMVA/CCPruner.h
  TMVA/CCTreeWrapper.h
  TMVA/Classification.h
  TMVA/ClassifierFactory.h
  TMVA/ClassInfo.h
  TMVA/Config.h
  TMVA/Configurable.h
  TMVA/ConvergenceTest.h
  TMVA/CostComplexityPruneTool.h
  TMVA/CrossEntropy.h
  TMVA/CrossValidation.h
  TMVA/CvSplit.h
  TMVA/DataInputHandler.h
  TMVA/DataLoader.h
  TMVA/DataSetFactory.h
  TMVA/DataSet.h
  TMVA/DataSetInfo.h
  TMVA/DataSetManager.h
  TMVA/DecisionTree.h
  TMVA/DecisionTreeNode.h
  TMVA/Envelope.h
  TMVA/Event.h
  TMVA/ExpectedErrorPruneTool.h
  TMVA/Factory.h
  TMVA/FitterBase.h
  TMVA/GeneticAlgorithm.h
  TMVA/GeneticFitter.h
  TMVA/GeneticGenes.h
  TMVA/GeneticPopulation.h
  TMVA/GeneticRange.h
  TMVA/GiniIndex.h
  TMVA/GiniIndexWithLaplace.h
  TMVA/HyperParameterOptimisation.h
  TMVA/IFitterTarget.h
  TMVA/IMethod.h
  TMVA/Interval.h
  TMVA/IPruneTool.h
  TMVA/KDEKernel.h
  TMVA/LDA.h
  TMVA/LogInterval.h
  TMVA/LossFunction.h
  TMVA/MCFitter.h
  TMVA/MethodANNBase.h
  TMVA/MethodBase.h
  TMVA/MethodBayesClassifier.h
  TMVA/MethodBDT.h
  TMVA/MethodBoost.h
  TMVA/MethodCategory.h
  TMVA/MethodCFMlpANN_def.h
  TMVA/MethodCFMlpANN.h
  TMVA/MethodCFMlpANN_Utils.h
  TMVA/MethodCompositeBase.h
  TMVA/MethodCrossValidation.h
  TMVA/MethodCuts.h
  TMVA/MethodDL.h
  TMVA/MethodDNN.h
  TMVA/MethodDT.h
  TMVA/MethodFDA.h
  TMVA/MethodFisher.h
  TMVA/MethodHMatrix.h
  TMVA/MethodKNN.h
  TMVA/MethodLD.h
  TMVA/MethodLikelihood.h
  TMVA/MethodMLP.h
  TMVA/MethodPDEFoam.h
  TMVA/MethodPDERS.h
  TMVA/MethodRuleFit.h
  TMVA/MethodSVM.h
  TMVA/MethodTMlpANN.h
  TMVA/MinuitFitter.h
  TMVA/MinuitWrapper.h
  TMVA/MisClassificationError.h
  TMVA/ModulekNN.h
  TMVA/Monitoring.h
  TMVA/MsgLogger.h
  TMVA/NeuralNet.h
  TMVA/Node.h
  TMVA/NodekNN.h
  TMVA/OptimizeConfigParameters.h
  TMVA/Option.h
  TMVA/OptionMap.h
  TMVA/Pattern.h
  TMVA/PDEFoamCell.h
  TMVA/PDEFoamDecisionTreeDensity.h
  TMVA/PDEFoamDecisionTree.h
  TMVA/PDEFoamDensityBase.h
  TMVA/PDEFoamDiscriminantDensity.h
  TMVA/PDEFoamDiscriminant.h
  TMVA/PDEFoamEventDensity.h
  TMVA/PDEFoamEvent.h
  TMVA/PDEFoam.h
  TMVA/PDEFoamKernelBase.h
  TMVA/PDEFoamKernelGauss.h
  TMVA/PDEFoamKernelLinN.h
  TMVA/PDEFoamKernelTrivial.h
  TMVA/PDEFoamMultiTarget.h
  TMVA/PDEFoamTargetDensity.h
  TMVA/PDEFoamTarget.h
  TMVA/PDEFoamVect.h
  TMVA/PDF.h
  TMVA/QuickMVAProbEstimator.h
  TMVA/Ranking.h
  TMVA/Reader.h
  TMVA/RegressionVariance.h
  TMVA/ResultsClassification.h
  TMVA/Results.h
  TMVA/ResultsMulticlass.h
  TMVA/ResultsRegression.h
  TMVA/ROCCalc.h
  TMVA/ROCCurve.h
  TMVA/RootFinder.h
  TMVA/RuleCut.h
  TMVA/RuleEnsemble.h
  TMVA/RuleFitAPI.h
  TMVA/RuleFit.h
  TMVA/RuleFitParams.h
  TMVA/Rule.h
  TMVA/SdivSqrtSplusB.h
  TMVA/SeparationBase.h
  TMVA/SimulatedAnnealingFitter.h
  TMVA/SimulatedAnnealing.h
  TMVA/SVEvent.h
  TMVA/SVKernelFunction.h
  TMVA/SVKernelMatrix.h
  TMVA/SVWorkingSet.h
  TMVA/TActivationChooser.h
  TMVA/TActivation.h
  TMVA/TActivationIdentity.h
  TMVA/TActivationRadial.h
  TMVA/TActivationReLU.h
  TMVA/TActivationSigmoid.h
  TMVA/TActivationTanh.h
  TMVA/Timer.h
  TMVA/TNeuron.h
  TMVA/TNeuronInputAbs.h
  TMVA/TNeuronInputChooser.h
  TMVA/TNeuronInput.h
  TMVA/TNeuronInputSqSum.h
  TMVA/TNeuronInputSum.h
  TMVA/Tools.h
  TMVA/TransformationHandler.h
  TMVA/TSpline1.h
  TMVA/TSpline2.h
  TMVA/TSynapse.h
  TMVA/Types.h
  TMVA/VariableDecorrTransform.h
  TMVA/VariableGaussTransform.h
  TMVA/VariableIdentityTransform.h
  TMVA/VariableImportance.h
  TMVA/VariableInfo.h
  TMVA/VariableNormalizeTransform.h
  TMVA/VariablePCATransform.h
  TMVA/VariableRearrangeTransform.h
  TMVA/VariableTransformBase.h
  TMVA/VariableTransform.h
  TMVA/VarTransformHandler.h
  TMVA/Version.h
  TMVA/Volume.h
)

#---Assign source files to the implementations -----------------
SET(DNN_FILES      src/DNN/Architectures/Reference.cxx
                   src/DNN/Architectures/Reference/DataLoader.cxx
                   src/DNN/Architectures/Reference/TensorDataLoader.cxx)
SET(DNN_CUDA_FILES src/DNN/Architectures/Cuda.cu
                   src/DNN/Architectures/Cuda/CudaBuffers.cxx
                   src/DNN/Architectures/Cuda/CudaMatrix.cu)
SET(DNN_CPU_FILES  src/DNN/Architectures/Cpu.cxx
                   src/DNN/Architectures/Cpu/CpuBuffer.cxx
                   src/DNN/Architectures/Cpu/CpuMatrix.cxx)

#---Handle CUDA dependent code. -----------------
if(CUDA_FOUND)
  message(STATUS "Using TMVA-DNN with CUDA installation") 
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fPIC")
  CUDA_INCLUDE_DIRECTORIES("${CMAKE_BINARY_DIR}/include")	
  CUDA_ADD_LIBRARY(dnn_cuda ${DNN_CUDA_FILES})
  set_property(TARGET dnn_cuda PROPERTY POSITION_INDEPENDENT_CODE ON)	
  set(DNN_CUDA_LIBRARIES dnn_cuda ${CUDA_CUBLAS_LIBRARIES})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNNCUDA")
  add_dependencies(dnn_cuda move_headers)
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} )
  message("Cuda include ${CUDA_INCLUDE_DIRS}")
else()
  set(DNN_CUDA_LIBRARIES)
  set(installoptions ${installoptions} FILTER "Cuda")
endif()

#---Handle BLAS dependent code. -----------------
if(BLAS_FOUND AND imt)
   message(STATUS "Using TMVA-DNN with BLAS installation: ${BLAS_LIBRARIES}") 
  set(DNN_CPU_LIBRARIES MathCore Matrix ${BLAS_LINKER_FLAGS} ${BLAS_LIBRARIES} ${TBB_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT}) 
  include_directories(${TBB_INCLUDE_DIRS})
  if (CUDA_FOUND)
     include_directories(${TBB_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
  endif()
else()
if (tmva-cpu AND mathmore AND imt)
#use GSL cblas installation
  message(STATUS "Using TMVA-DNN with gslcblas installation") 
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDNN_USE_CBLAS")
  set(DNN_CPU_LIBRARIES MathCore Matrix ${TBB_LIBRARIES} ${GSL_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
  include_directories(${TBB_INCLUDE_DIRS} ${GSL_INCLUDE_DIR} )
  if (CUDA_FOUND)
     include_directories(${TBB_INCLUDE_DIRS} ${GSL_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
  endif()
else()
  message(STATUS "No blas or cblas found . TMVA-DNN-CPU is disabled")  
  set(DNN_CPU_LIBRARIES)
  set(DNN_CPU_FILES)
  set(installoptions ${installoptions} FILTER "Cpu")
endif()
endif()

if(imt)
  set(TMVA_DEPENDENCIES Imt)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(TMVA
                              HEADERS ${TMVA_HEADERS}
                              SOURCES *.cxx ${DNN_FILES} ${DNN_CPU_FILES} DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              LIBRARIES Core MathCore Matrix ${DNN_CUDA_LIBRARIES} ${DNN_CPU_LIBRARIES}
                              DEPENDENCIES Core RIO Hist Tree TreePlayer MLP Minuit XMLIO ${TMVA_DEPENDENCIES}
                              INSTALL_OPTIONS ${installoptions})

ROOT_ADD_TEST_SUBDIRECTORY(test)
ROOT_ADD_TEST_SUBDIRECTORY(test/crossvalidation)
ROOT_ADD_TEST_SUBDIRECTORY(test/DNN)
ROOT_ADD_TEST_SUBDIRECTORY(test/Method)
ROOT_ADD_TEST_SUBDIRECTORY(test/ROC)
ROOT_ADD_TEST_SUBDIRECTORY(test/envelope)
ROOT_ADD_TEST_SUBDIRECTORY(test/DNN/CNN)
ROOT_ADD_TEST_SUBDIRECTORY(test/DNN/RNN)
