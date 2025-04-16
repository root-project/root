\addtogroup tutorial_ml


## Table of contents
- [Basic TMVA tutorials](\ref basic)
   - [Training](\ref training)
   - [Applications](\ref application)
   - [Others](\ref other)
- [Cross validation](\ref cross_val)
- [New TMVA interfaces](\ref new_interface)
   - [RBDT](\ref rbdt)
- [Deep learning in TMVA](\ref deep_learing)
- [TMVA Keras tutorials](\ref keras)
- [TMVA PyTorch tutorials](\ref pytorch)
- [Interference with SOFIE](\ref interference)         
- [RBatchGenerator](\ref rbatchgen)


\anchor basic
## Basic TMVA tutorials


\anchor training
### Training

| **Tutorial** | **Description** |
|--------------|-----------------|
| TMVAMinimalClassification.C |  Minimal self-contained example for setting up TMVA with binary classification. |
| TMVAMulticlass.C | This macro provides a simple example for the training and testing of the TMVA multiclass classification. |
| TMVAClassification.C |  This macro provides examples for the training and testing of the TMVA classifiers. |
| TMVAClassificationCategory.C | This macro provides examples for the training and testing of the TMVA classifiers in categorisation mode. |
| TMVARegression.C | This macro provides examples for the training and testing of the TMVA classifiers. |
| classification.C | |

\anchor application
### Applications

| **Tutorial** | **Description** |
|--------------|-----------------|
| TMVAMulticlassApplication.C | This macro provides a simple example on how to use the trained multiclass classifiers within an analysis module. |
| TMVAClassificationApplication.C | This macro provides a simple example on how to use the trained classifiers within an analysis module. |
| TMVAClassificationCategoryApplication.C | This macro provides a simple example on how to use the trained classifiers (with categories) within an analysis module. |
| TMVACrossValidationApplication.C | This macro provides an example of how to use TMVA for k-folds cross evaluation in application. |
| TMVAMulticlassApplication.C | This macro provides a simple example on how to use the trained multiclass classifiers within an analysis module. |
| TMVARegressionApplication.C | This macro provides a simple example on how to use the trained regression MVAs within an analysis module. |

\anchor other
### Others

| **Tutorial** | **Description** |
|--------------|-----------------|
| TMVAGAexample.C | This executable gives an example of a very simple use of the genetic algorithm of TMVA. |
| TMVAGAexample2.C | This executable gives an example of a very simple use of the genetic algorithm of TMVA. |
| TMVAMultipleBackgroundExample.C | This example shows the training of signal with three different backgrounds. |

\anchor cross_val
## Cross validation
| **Tutorial** | **Description** |
|--------------|-----------------|
| TMVACrossValidation.C | This macro provides an example of how to use TMVA for k-folds cross evaluation. |
| TMVACrossValidationApplication.C | This macro provides an example of how to use TMVA for k-folds cross evaluation in application. |
| TMVACrossValidationRegression.C | This macro provides an example of how to use TMVA for k-folds cross evaluation. |

\anchor new_interface
## New TMVA interfaces

| **Tutorial** | **Description** |
|--------------|-----------------|
| createData.C | Plot the variables. |
| tmva001_RTensor.C | This tutorial illustrates the basic features of the RTensor class, RTensor is a std::vector-like container with additional shape information. |
| tmva002_RDataFrameAsTensor.C |  This tutorial shows how the content of an RDataFrame can be converted to an RTensor object. |
| tmva003_RReader.C |  This tutorial shows how to apply with the modern interfaces models saved in TMVA XML files. |


\anchor rbdt
### RBDT
| **Tutorial** | **Description** |
|--------------|-----------------|
| tmva100_DataPreparation.py | This tutorial illustrates how to prepare ROOT datasets to be nicely readable by most machine learning methods. |
| tmva101_Training.py | This tutorial show how you can train a machine learning model with any package reading the training data directly from ROOT files. |
| tmva102_Testing.py |  This tutorial illustrates how you can test a trained BDT model using the fast tree inference engine offered by TMVA and external tools such as scikit-learn.  |
| tmva103_Application.C |  This tutorial illustrates how you can conveniently apply BDTs in C++ using the fast tree inference engine offered by TMVA. |

\anchor deep_learing
## Deep learning in TMVA

|          **Tutorial**          || **Description** |
|---------------|-----------------|-----------------|
| TMVA_CNN_Classification.C | TMVA_CNN_Classification.py |  TMVA Classification Example Using a Convolutional Neural Network. |
| TMVA_Higgs_Classification.C | TMVA_Higgs_Classification.py |  Classification example of TMVA based on public Higgs UCI dataset. |
| TMVA_RNN_Classification.C | TMVA_RNN_Classification.py |  TMVA Classification Example Using a Recurrent Neural Network. |

\anchor keras
## TMVA Keras tutorials

| **Tutorial** | **Description** |
|--------------|-----------------|
| ApplicationClassificationKeras.py | This tutorial shows how to apply a trained model to new data. |
| ApplicationRegressionKeras.py |  This tutorial shows how to apply a trained model to new data (regression). | 
| ClassificationKeras.py |  This tutorial shows how to do classification in TMVA with neural networks trained with keras. |
| GenerateModel.py |  This tutorial shows how to define and generate a keras model for use with TMVA. | 
| MulticlassKeras.py |  This tutorial shows how to do multiclass classification in TMVA with neural networks trained with keras. |
| RegressionKeras.py |  This tutorial shows how to do regression in TMVA with neural networks trained with keras. |

\anchor pytorch
## TMVA PyTorch tutorials
| **Tutorial** | **Description** |
|--------------|-----------------|
| ApplicationClassificationPyTorch.py |  This tutorial shows how to apply a trained model to new data. |
| ApplicationRegressionPyTorch.py |  This tutorial shows how to apply a trained model to new data (regression). |
| ClassificationPyTorch.py |  This tutorial shows how to do classification in TMVA with neural networks trained with PyTorch. |
| MulticlassPyTorch.py |  This tutorial shows how to do multiclass classification in TMVA with neural networks trained with PyTorch. |
| RegressionPyTorch.py |  This tutorial shows how to do regression in TMVA with neural networks trained with PyTorch. |



\anchor interference
## Interference with SOFIE

|          **Tutorial**          || **Description** |
|---------------|-----------------|-----------------|
| | TMVA_SOFIE_Inference.py | This macro provides an example of using a trained model with Keras and make inference using SOFIE directly from Numpy. |
| TMVA_SOFIE_Keras.C | | This macro provides a simple example for the parsing of Keras .h5 file into RModel object and further generating the .hxx header files for inference. |
| TMVA_SOFIE_Keras_HiggsModel.C | | This macro run the SOFIE parser on the Keras model obtaining running TMVA_Higgs_Classification.C You need to run that macro before this one. |
| | TMVA_SOFIE_Models.py |  Example of inference with SOFIE using a set of models trained with Keras. |
| TMVA_SOFIE_ONNX.C | | This macro provides a simple example for the parsing of ONNX files into RModel object and further generating the .hxx header files for inference. |
| TMVA_SOFIE_PyTorch.C | |  This macro provides a simple example for the parsing of PyTorch .pt file into RModel object and further generating the .hxx header files for inference. |
| TMVA_SOFIE_RDataFrame.C | TMVA_SOFIE_RDataFrame.py |  Example of inference with SOFIE and RDataFrame, of a model trained with Keras. |
| TMVA_SOFIE_RDataFrame_JIT.C | | This macro provides an example of using a trained model with Keras and make inference using SOFIE and RDataFrame. |
| TMVA_SOFIE_RSofieReader.C | | This macro provides an example of using a trained model with Keras and make inference using SOFIE with the RSofieReader class. |

\anchor rbatchgen
## RBatchGenerator

| **Tutorial** | **Description** |
|--------------|-----------------|
| RBatchGenerator_NumPy.py | Example of getting batches of events from a ROOT dataset as Python generators of numpy arrays. |
| RBatchGenerator_PyTorch.py | Example of getting batches of events from a ROOT dataset into a basic PyTorch workflow. |
| RBatchGenerator_TensorFlow.py | Example of getting batches of events from a ROOT dataset into a basic TensorFlow workflow. |

