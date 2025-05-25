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
- [Inference with SOFIE](\ref inference)         
- [Data loading for training](\ref data_loading)


\anchor basic
## Basic TMVA tutorials


\anchor training
### Training

| **Tutorial** | **Description** |
|--------------|-----------------|
| TMVAMinimalClassification.C |  Minimal self-contained example for setting up TMVA with binary classification. |
| TMVAMulticlass.C | Training and testing of the TMVA multiclass classification. |
| TMVAClassification.C |  Training and testing of the TMVA classifiers. |
| TMVAClassificationCategory.C | Training and testing of the TMVA classifiers in categorisation mode. |
| TMVARegression.C | Training and testing of the TMVA classifiers. |
| classification.C | |

\anchor application
### Applications

| **Tutorial** | **Description** |
|--------------|-----------------|
| TMVAMulticlassApplication.C | Using the trained multiclass classifiers within an analysis module. |
| TMVAClassificationApplication.C | Using the trained classifiers within an analysis module. |
| TMVAClassificationCategoryApplication.C | Using the trained classifiers (with categories) within an analysis module. |
| TMVACrossValidationApplication.C | Using TMVA for k-folds cross evaluation in application. |
| TMVAMulticlassApplication.C | Using trained multiclass classifiers within an analysis module. |
| TMVARegressionApplication.C | Using the trained regression MVAs within an analysis module. |

\anchor other
### Others

| **Tutorial** | **Description** |
|--------------|-----------------|
| TMVAGAexample.C | Using the genetic algorithm of TMVA. |
| TMVAGAexample2.C | Using the genetic algorithm of TMVA. |
| TMVAMultipleBackgroundExample.C | Training of signal with three different backgrounds. |

\anchor cross_val
## Cross validation
| **Tutorial** | **Description** |
|--------------|-----------------|
| TMVACrossValidation.C | Using the TMVA k-folds cross evaluation. |
| TMVACrossValidationApplication.C | Using the TMVA k-folds cross evaluation in application. |
| TMVACrossValidationRegression.C | Using the TMVA k-folds cross evaluation. |

\anchor new_interface
## New TMVA interfaces

| **Tutorial** | **Description** |
|--------------|-----------------|
| createData.C | Plot the variables. |
| tmva001_RTensor.C | Illustrate the basic features of the RTensor class, RTensor is a std::vector-like container with additional shape information. |
| tmva002_RDataFrameAsTensor.C | Convert the content of an RDataFrame to an RTensor object. |
| tmva003_RReader.C | Use modern interfaces models saved in TMVA XML files. |


\anchor rbdt
### RBDT
| **Tutorial** | **Description** |
|--------------|-----------------|
| tmva100_DataPreparation.py | Prepare ROOT datasets to be nicely readable by most machine learning methods. |
| tmva101_Training.py | Train a machine learning model with any package reading the training data directly from ROOT files. |
| tmva102_Testing.py | Test a trained BDT model using the fast tree inference engine offered by TMVA and external tools such as scikit-learn.  |
| tmva103_Application.C | Apply BDTs in C++ using the fast tree inference engine offered by TMVA. |

\anchor deep_learing
## Deep learning in TMVA

|          **Tutorial**          || **Description** |
|---------------|-----------------|-----------------|
| TMVA_CNN_Classification.C | TMVA_CNN_Classification.py |  TMVA Classification example using a Convolutional Neural Network. |
| TMVA_Higgs_Classification.C | TMVA_Higgs_Classification.py |  Classification example of TMVA based on public Higgs UCI dataset. |
| TMVA_RNN_Classification.C | TMVA_RNN_Classification.py |  TMVA Classification example using a Recurrent Neural Network. |

\anchor keras
## TMVA Keras tutorials

| **Tutorial** | **Description** |
|--------------|-----------------|
| ApplicationClassificationKeras.py | Apply a trained model to new data. |
| ApplicationRegressionKeras.py |  Apply a trained model to new data (regression). | 
| ClassificationKeras.py |  Classification in TMVA with neural networks trained with keras. |
| GenerateModel.py |  Define and generate a keras model for use with TMVA. | 
| MulticlassKeras.py |  Multiclass classification in TMVA with neural networks trained with keras. |
| RegressionKeras.py | Regression in TMVA with neural networks trained with keras. |

\anchor pytorch
## TMVA PyTorch tutorials
| **Tutorial** | **Description** |
|--------------|-----------------|
| ApplicationClassificationPyTorch.py | Apply a trained model to new data. |
| ApplicationRegressionPyTorch.py |  Apply a trained model to new data (regression). |
| ClassificationPyTorch.py | Classification in TMVA with neural networks trained with PyTorch. |
| MulticlassPyTorch.py | Multiclass classification in TMVA with neural networks trained with PyTorch. |
| RegressionPyTorch.py | Regression in TMVA with neural networks trained with PyTorch. |



\anchor inference
## Inference with SOFIE

|          **Tutorial**          || **Description** |
|---------------|-----------------|-----------------|
| | TMVA_SOFIE_Inference.py | Using a trained model with Keras and make inference using SOFIE directly from Numpy. |
| TMVA_SOFIE_Keras.C | | Parsing of Keras .h5 file into RModel object and further generating the .hxx header files for inference. |
| TMVA_SOFIE_Keras_HiggsModel.C | | Run the SOFIE parser on the Keras model obtaining running TMVA_Higgs_Classification.C. You need to run that macro before this one. |
| | TMVA_SOFIE_Models.py | Inference with SOFIE using a set of models trained with Keras. |
| TMVA_SOFIE_ONNX.C | | Parsing of ONNX files into RModel object and further generating the .hxx header files for inference. |
| TMVA_SOFIE_PyTorch.C | | Parsing of PyTorch .pt file into RModel object and further generating the .hxx header files for inference. |
| TMVA_SOFIE_RDataFrame.C | TMVA_SOFIE_RDataFrame.py | Inference with SOFIE and RDataFrame, of a model trained with Keras. |
| TMVA_SOFIE_RDataFrame_JIT.C | | Using a trained model with Keras and make inference using SOFIE and RDataFrame. |
| TMVA_SOFIE_RSofieReader.C | | Using a trained model with Keras and make inference using SOFIE with the RSofieReader class. |

\anchor data_loading
## Data loading for training

| **Tutorial** | **Description** |
|--------------|-----------------|
| RBatchGenerator_NumPy.py | Loading batches of events from a ROOT dataset as Python generators of numpy arrays. |
| RBatchGenerator_PyTorch.py | Loading batches of events from a ROOT dataset into a basic PyTorch workflow. |
| RBatchGenerator_TensorFlow.py | Loading batches of events from a ROOT dataset into a basic TensorFlow workflow. |

