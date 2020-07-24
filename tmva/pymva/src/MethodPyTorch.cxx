// @(#)root/tmva/pymva $Id$
// Author: Anirudh Dagar, 2020

#include <Python.h>
#include "TMVA/MethodPyTorch.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "TMVA/Types.h"
#include "TMVA/Config.h"
#include "TMVA/ClassifierFactory.h"
#include "TMVA/Results.h"
#include "TMVA/TransformationHandler.h"
#include "TMVA/VariableTransformBase.h"
#include "TMVA/Tools.h"
#include "TMVA/Timer.h"

using namespace TMVA;

namespace TMVA {
namespace Internal {
class PyGILRAII {
   PyGILState_STATE m_GILState;

public:
   PyGILRAII() : m_GILState(PyGILState_Ensure()) {}
   ~PyGILRAII() { PyGILState_Release(m_GILState); }
};
} // namespace Internal
} // namespace TMVA

REGISTER_METHOD(PyTorch)

ClassImp(MethodPyTorch);

MethodPyTorch::MethodPyTorch(const TString &jobName, const TString &methodTitle, DataSetInfo &dsi, const TString &theOption)
   : PyMethodBase(jobName, Types::kPyTorch, methodTitle, dsi, theOption) {
   fNumEpochs = 10;
   fBatchSize = 100;
   
   // TODO: Ignore if verbosity not required in pytorch models.
   fVerbose = 1;

   fContinueTraining = false;
   fSaveBestOnly = true;
   fTriesEarlyStopping = -1;
   fLearningRateSchedule = ""; // empty string deactivates learning rate scheduler
   fFilenameTrainedModel = ""; // empty string sets output model filename to default (in "weights/" directory.)
   fTensorBoard = "";          // empty string deactivates TensorBoard
}

MethodPyTorch::MethodPyTorch(DataSetInfo &theData, const TString &theWeightFile)
    : PyMethodBase(Types::kPyTorch, theData, theWeightFile) {
   fNumEpochs = 10;
   fBatchSize = 100;

   // TODO: Ignore if verbosity not required in pytorch models.
   fVerbose = 1;
   
   fContinueTraining = false;
   fSaveBestOnly = true;
   fTriesEarlyStopping = -1;
   fLearningRateSchedule = ""; // empty string deactivates learning rate scheduler
   fFilenameTrainedModel = ""; // empty string sets output model filename to default (in weights/)
   fTensorBoard = "";          // empty string deactivates TensorBoard
}

MethodPyTorch::~MethodPyTorch() {
}

Bool_t MethodPyTorch::HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t) {
   if (type == Types::kRegression) return kTRUE;
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   if (type == Types::kMulticlass && numberClasses >= 2) return kTRUE;
   return kFALSE;
}

///////////////////////////////////////////////////////////////////////////////

void MethodPyTorch::DeclareOptions() {
   DeclareOptionRef(fFilenameModel, "FilenameModel", "Filename of the initial PyTorch model");
   DeclareOptionRef(fFilenameTrainedModel, "FilenameTrainedModel", "Filename of the trained output PyTorch model");
   DeclareOptionRef(fBatchSize, "BatchSize", "Training batch size");
   DeclareOptionRef(fNumEpochs, "NumEpochs", "Number of training epochs");

   // TODO: Check if verbosity is required in pytorch models.
   DeclareOptionRef(fVerbose, "Verbose", "PyTorch verbosity during training");

   DeclareOptionRef(fContinueTraining, "ContinueTraining", "Load weights from previous training");
   DeclareOptionRef(fSaveBestOnly, "SaveBestOnly", "Store only weights with smallest validation loss");
   DeclareOptionRef(fTriesEarlyStopping, "TriesEarlyStopping", "Number of epochs with no improvement in validation loss after "
                                          "which training will be stopped. The default or a negative number deactivates this option.");
   DeclareOptionRef(fLearningRateSchedule, "LearningRateSchedule", "Set new learning rate during training at specific epochs, e.g., \"50,0.01;70,0.005\" using torch.optim.lr_scheduler");
   DeclareOptionRef(fTensorBoard, "TensorBoard",
                    "Write a log during training to visualize and monitor the training performance with torch.utils.tensorboard");

   DeclareOptionRef(fNumValidationString = "20%", "ValidationSize", "Part of the training data to use for validation."
                    "Specify as 0.2 or 20% to use a fifth of the data set as validation set."
                    "Specify as 100 to use exactly 100 events. (Default: 20%)");
   DeclareOptionRef(fUserCodeName = "", "UserCode", "Necessary python code provided by the user to be executed before loading and training the PyTorch Model");

}


////////////////////////////////////////////////////////////////////////////////
/// Validation of the ValidationSize option. Allowed formats are 20%, 0.2 and
/// 100 etc.
///    - 20% and 0.2 selects 20% of the training set as validation data.
///    - 100 selects 100 events as the validation data.
///
/// @return number of samples in validation set
///
UInt_t TMVA::MethodPyTorch::GetNumValidationSamples()
{
   Int_t nValidationSamples = 0;
   UInt_t trainingSetSize = GetEventCollection(Types::kTraining).size();

   // Parsing + Validation
   // --------------------
   if (fNumValidationString.EndsWith("%")) {
      // Relative spec. format 20%
      TString intValStr = TString(fNumValidationString.Strip(TString::kTrailing, '%'));

      if (intValStr.IsFloat()) {
         Double_t valSizeAsDouble = fNumValidationString.Atof() / 100.0;
         nValidationSamples = GetEventCollection(Types::kTraining).size() * valSizeAsDouble;
      } else {
         Log() << kFATAL << "Cannot parse number \"" << fNumValidationString
               << "\". Expected string like \"20%\" or \"20.0%\"." << Endl;
      }
   } else if (fNumValidationString.IsFloat()) {
      Double_t valSizeAsDouble = fNumValidationString.Atof();

      if (valSizeAsDouble < 1.0) {
         // Relative spec. format 0.2
         nValidationSamples = GetEventCollection(Types::kTraining).size() * valSizeAsDouble;
      } else {
         // Absolute spec format 100 or 100.0
         nValidationSamples = valSizeAsDouble;
      }
   } else {
      Log() << kFATAL << "Cannot parse number \"" << fNumValidationString << "\". Expected string like \"0.2\" or \"100\"."
            << Endl;
   }

   // Value validation
   // ----------------
   if (nValidationSamples < 0) {
      Log() << kFATAL << "Validation size \"" << fNumValidationString << "\" is negative." << Endl;
   }

   if (nValidationSamples == 0) {
      Log() << kFATAL << "Validation size \"" << fNumValidationString << "\" is zero." << Endl;
   }

   if (nValidationSamples >= (Int_t)trainingSetSize) {
      Log() << kFATAL << "Validation size \"" << fNumValidationString
            << "\" is larger than or equal in size to training set (size=\"" << trainingSetSize << "\")." << Endl;
   }

   return nValidationSamples;
}

void MethodPyTorch::ProcessOptions() {
   // Set default filename for trained model if option is not used
   if (fFilenameTrainedModel.IsNull()) {
      fFilenameTrainedModel = GetWeightFileDir() + "/TrainedModel_" + GetName() + ".pt";
   }

   //  -  set up number of threads for CPU if NumThreads option was specified
   //     `torch.set_num_threads` sets the number of threads that can be used to
   //     perform cpu operations like conv or mm (usually used by OpenMP or MKL).

   Log() << kINFO << "Using PyTorch - setting special configuration options "  << Endl;
   PyRunString("import torch", "Error importing pytorch");

      // run these above lines also in global namespace to make them visible overall
   PyRun_String("import torch", Py_single_input, fGlobalNS, fGlobalNS);

   // check pytorch version
   PyRunString("torch_major_version = int(torch.__version__.split('.')[0])");
   PyObject *pyTorchVersion = PyDict_GetItemString(fLocalNS, "torch_major_version");
   int torchVersion = PyLong_AsLong(pyTorchVersion);
   Log() << kINFO << "Using PyTorch version " << torchVersion << Endl;

   // in case specify number of threads
   int num_threads = fNumThreads;
   if (num_threads > 0) {
      Log() << kINFO << "Setting the CPU number of threads =  "  << num_threads << Endl;

      PyRunString(TString::Format("torch.set_num_threads(%d)", num_threads));
      PyRunString(TString::Format("torch.set_num_interop_threads(%d)", num_threads));
   }

   // Setup model, either the initial model from `fFilenameModel` or
   // the trained model from `fFilenameTrainedModel`
   if (fContinueTraining) Log() << kINFO << "Continue training with trained model" << Endl;
   SetupPyTorchModel(fContinueTraining);
}


void MethodPyTorch::SetupPyTorchModel(bool loadTrainedModel) {
   /*
    * Load PyTorch model from file
    */

   Log() << kINFO << " Setup PyTorch Model " << Endl;

   PyRunString("load_model_custom_objects=None");


   if (!fUserCodeName.IsNull()) {
      Log() << kINFO << " Executing user initialization code from  " << fUserCodeName << Endl;


      // run some python code provided by user for model initialization if needed
      TString cmd = "exec(open('" + fUserCodeName + "').read())";
      TString errmsg = "Error executing the provided user code";
      PyRunString(cmd, errmsg);

      PyRunString("print('custom objects for loading model : ',load_model_custom_objects)");

      PyRunString("fit = load_model_custom_objects[\"train_func\"]",
                  "Failed to load train function from file");
      Log() << kINFO << "Loaded pytorch train function: " << Endl;


      PyRunString("optimizer = load_model_custom_objects[\"optimizer\"]",
                  "Failed to load optimizer object from file");
      Log() << kINFO << "Loaded pytorch optimizer: " << Endl;


      PyRunString("train = load_model_custom_objects[\"criterion\"]",
                  "Failed to load loss function from file");
      Log() << kINFO << "Loaded pytorch loss function: " << Endl;
   }


   // Load already trained model or initial model
   TString filenameLoadModel;
   if (loadTrainedModel) {
      filenameLoadModel = fFilenameTrainedModel;
   }
   else {
      filenameLoadModel = fFilenameModel;
   }
   PyRunString("model = torch.jit.load('"+filenameLoadModel+"')",
               "Failed to load PyTorch model from file: "+filenameLoadModel);
   Log() << kINFO << "Load model from file: " << filenameLoadModel << Endl;


   /*
    * Init variables and weights
    */

   // Get variables, classes and target numbers
   fNVars = GetNVariables();
   if (GetAnalysisType() == Types::kClassification || GetAnalysisType() == Types::kMulticlass) fNOutputs = DataInfo().GetNClasses();
   else if (GetAnalysisType() == Types::kRegression) fNOutputs = DataInfo().GetNTargets();
   else Log() << kFATAL << "Selected analysis type is not implemented" << Endl;

   // Init evaluation (needed for getMvaValue)
   fVals = new float[fNVars]; // holds values used for classification and regression
   npy_intp dimsVals[2] = {(npy_intp)1, (npy_intp)fNVars};
   PyArrayObject* pVals = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimsVals, NPY_FLOAT, (void*)fVals);
   PyDict_SetItemString(fLocalNS, "vals", (PyObject*)pVals);

   fOutput.resize(fNOutputs); // holds classification probabilities or regression output
   npy_intp dimsOutput[2] = {(npy_intp)1, (npy_intp)fNOutputs};
   PyArrayObject* pOutput = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimsOutput, NPY_FLOAT, (void*)&fOutput[0]);
   PyDict_SetItemString(fLocalNS, "output", (PyObject*)pOutput);

   // Mark the model as setup
   fModelIsSetup = true;
}

void MethodPyTorch::Init() {

   TMVA::Internal::PyGILRAII raii;

   if (!PyIsInitialized()) {
      Log() << kFATAL << "Python is not initialized" << Endl;
   }
   _import_array(); // required to use numpy arrays

   // Import PyTorch
   // TODO: Skip clearing sys.argv for pytorch
   PyRunString("import sys; sys.argv = ['']", "Set sys.argv failed");
   PyRunString("import torch", "import PyTorch failed");
   // do import also in global namespace
   auto ret = PyRun_String("import torch", Py_single_input, fGlobalNS, fGlobalNS);
   if (!ret)
      Log() << kFATAL << "import PyTorch in global namespace failed " << Endl;

   // Set flag that model is not setup
   fModelIsSetup = false;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO: This method is not fully implemented yet. Callback specific structure in PyTorch TBD. 
// Adaptation in process from PyKeras

/////////////////////////////////////////////////////////////////////////////////////////////////// 

void MethodPyTorch::Train() {
   if(!fModelIsSetup) Log() << kFATAL << "Model is not setup for training" << Endl;

   /*
    * Load training data to numpy array. 
    * NOTE: These are later forced to be converted into torch tensors throught the training loop which may not be the ideal method.
    */

   UInt_t nAllEvents = Data()->GetNTrainingEvents();
   UInt_t nValEvents = GetNumValidationSamples();
   UInt_t nTrainingEvents = nAllEvents - nValEvents;

   Log() << kINFO << "Split TMVA training data in " << nTrainingEvents << " training events and "
         << nValEvents << " validation events" << Endl;

   float* trainDataX = new float[nTrainingEvents*fNVars];
   float* trainDataY = new float[nTrainingEvents*fNOutputs];
   float* trainDataWeights = new float[nTrainingEvents];
   for (UInt_t i=0; i<nTrainingEvents; i++) {
      const TMVA::Event* e = GetTrainingEvent(i);
      // Fill variables
      for (UInt_t j=0; j<fNVars; j++) {
         trainDataX[j + i*fNVars] = e->GetValue(j);
      }
      // Fill targets
      // NOTE: For classification, convert class number in one-hot vector,
      // e.g., 1 -> [0, 1] or 0 -> [1, 0] for binary classification
      if (GetAnalysisType() == Types::kClassification || GetAnalysisType() == Types::kMulticlass) {
         for (UInt_t j=0; j<fNOutputs; j++) {
            trainDataY[j + i*fNOutputs] = 0;
         }
         trainDataY[e->GetClass() + i*fNOutputs] = 1;
      }
      else if (GetAnalysisType() == Types::kRegression) {
         for (UInt_t j=0; j<fNOutputs; j++) {
            trainDataY[j + i*fNOutputs] = e->GetTarget(j);
         }
      }
      else Log() << kFATAL << "Can not fill target vector because analysis type is not known" << Endl;
      // Fill weights
      // NOTE: If no weight branch is given, this defaults to ones for all events
      trainDataWeights[i] = e->GetWeight();
   }

   npy_intp dimsTrainX[2] = {(npy_intp)nTrainingEvents, (npy_intp)fNVars};
   npy_intp dimsTrainY[2] = {(npy_intp)nTrainingEvents, (npy_intp)fNOutputs};
   npy_intp dimsTrainWeights[1] = {(npy_intp)nTrainingEvents};
   PyArrayObject* pTrainDataX = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimsTrainX, NPY_FLOAT, (void*)trainDataX);
   PyArrayObject* pTrainDataY = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimsTrainY, NPY_FLOAT, (void*)trainDataY);
   PyArrayObject* pTrainDataWeights = (PyArrayObject*)PyArray_SimpleNewFromData(1, dimsTrainWeights, NPY_FLOAT, (void*)trainDataWeights);
   PyDict_SetItemString(fLocalNS, "trainX", (PyObject*)pTrainDataX);
   PyDict_SetItemString(fLocalNS, "trainY", (PyObject*)pTrainDataY);
   PyDict_SetItemString(fLocalNS, "trainWeights", (PyObject*)pTrainDataWeights);

   /*
    * Load validation data to numpy array
    */

   // NOTE: TMVA Validation data is a subset of all the training data
   // we will not use test data for validation. They will be used for the real testing


   float* valDataX = new float[nValEvents*fNVars];
   float* valDataY = new float[nValEvents*fNOutputs];
   float* valDataWeights = new float[nValEvents];
   //validation events follows the trainig one in the TMVA training vector
   for (UInt_t i=0; i< nValEvents ; i++) {
      UInt_t ievt = nTrainingEvents + i; // TMVA event index
      const TMVA::Event* e = GetTrainingEvent(ievt);
      // Fill variables
      for (UInt_t j=0; j<fNVars; j++) {
         valDataX[j + i*fNVars] = e->GetValue(j);
      }
      // Fill targets
      if (GetAnalysisType() == Types::kClassification || GetAnalysisType() == Types::kMulticlass) {
         for (UInt_t j=0; j<fNOutputs; j++) {
            valDataY[j + i*fNOutputs] = 0;
         }
         valDataY[e->GetClass() + i*fNOutputs] = 1;
      }
      else if (GetAnalysisType() == Types::kRegression) {
         for (UInt_t j=0; j<fNOutputs; j++) {
            valDataY[j + i*fNOutputs] = e->GetTarget(j);
         }
      }
      else Log() << kFATAL << "Can not fill target vector because analysis type is not known" << Endl;
      // Fill weights
      valDataWeights[i] = e->GetWeight();
   }

   npy_intp dimsValX[2] = {(npy_intp)nValEvents, (npy_intp)fNVars};
   npy_intp dimsValY[2] = {(npy_intp)nValEvents, (npy_intp)fNOutputs};
   npy_intp dimsValWeights[1] = {(npy_intp)nValEvents};
   PyArrayObject* pValDataX = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimsValX, NPY_FLOAT, (void*)valDataX);
   PyArrayObject* pValDataY = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimsValY, NPY_FLOAT, (void*)valDataY);
   PyArrayObject* pValDataWeights = (PyArrayObject*)PyArray_SimpleNewFromData(1, dimsValWeights, NPY_FLOAT, (void*)valDataWeights);
   PyDict_SetItemString(fLocalNS, "valX", (PyObject*)pValDataX);
   PyDict_SetItemString(fLocalNS, "valY", (PyObject*)pValDataY);
   PyDict_SetItemString(fLocalNS, "valWeights", (PyObject*)pValDataWeights);

   /*
    * Train PyTorch model
    */
   Log() << kINFO << "Print Training Model Architecture" << Endl;
   PyRunString("print(model)");

   // Setup parameters

   PyObject* pBatchSize = PyLong_FromLong(fBatchSize);
   PyObject* pNumEpochs = PyLong_FromLong(fNumEpochs);
   // Ignore if verbosity is not required in pytorch models.
   PyObject* pVerbose = PyLong_FromLong(fVerbose);
   PyDict_SetItemString(fLocalNS, "batchSize", pBatchSize);
   PyDict_SetItemString(fLocalNS, "numEpochs", pNumEpochs);
   // TODO: Check if verbosity is required in pytorch models.
   PyDict_SetItemString(fLocalNS, "verbose", pVerbose);

   // ////////////////////////////////////////////////////////////////////
   // TODO: Better strategy for using Callbacks in PyTorch

   // Setup training functions acting as callbacks

   // SAVE BEST MODEL Save only weights with smallest validation loss

   // EARLY STOPPING Stop training early if no improvement in validation loss is observed

   // LRScheduler

   // PyTorch TensorBoard

   // ////////////////////////////////////////////////////////////////////

   // Train model
   PyRunString("fit(model, trainX, trainY, num_epochs=numEpochs, batch_size=batchSize, optimizer=optimizer, criterion=criterion)",
               "Failed to train model");


   // TODO: Add method to store training history data. 


   /*
    * Store trained model to file (only if option 'SaveBestOnly' is NOT activated,
    * because we do not want to override the best model checkpoint)
    */
   if (!fSaveBestOnly) {
      PyRunString("torch.jit.save('"+fFilenameTrainedModel+"')",
                  "Failed to save trained model: "+fFilenameTrainedModel);
      Log() << kINFO << "Trained model written to file: " << fFilenameTrainedModel << Endl;
   }

   /*
    * Clean-up
    */

   delete[] trainDataX;
   delete[] trainDataY;
   delete[] trainDataWeights;
   delete[] valDataX;
   delete[] valDataY;
   delete[] valDataWeights;
}

void MethodPyTorch::TestClassification() {
    MethodBase::TestClassification();
}

Double_t MethodPyTorch::GetMvaValue(Double_t *errLower, Double_t *errUpper) {
   // Cannot determine error
   NoErrorCalc(errLower, errUpper);

   // Check whether the model is setup
   // NOTE: unfortunately this is needed because during evaluation ProcessOptions is not called again
   if (!fModelIsSetup) {
      // Setup the trained model
      SetupPyTorchModel(true);
   }


   /////////////////////
   // TODO: PyTorch doesn't have a method to use direct predictions.
   //       Convert or omit to get the Keras like predictions setup.
   //       Also, set model to model.eval() before making predictions.
   /////////////////////


   // // Get signal probability (called mvaValue here)
   // const TMVA::Event* e = GetEvent();
   // for (UInt_t i=0; i<fNVars; i++) fVals[i] = e->GetValue(i);
   // PyRunString("model.eval(); for i,p in enumerate(model.predict(vals)): output[i]=p\n",
   //             "Failed to get predictions");


   return fOutput[TMVA::Types::kSignal];
}


std::vector<Double_t> MethodPyTorch::GetMvaValues(Long64_t firstEvt, Long64_t lastEvt, Bool_t logProgress) {
   // Check whether the model is setup
   // NOTE: Unfortunately this is needed because during evaluation ProcessOptions is not called again
   if (!fModelIsSetup) {
      // Setup the trained model
      SetupPyTorchModel(true);
   }

   // Load data to numpy array
   Long64_t nEvents = Data()->GetNEvents();
   if (firstEvt > lastEvt || lastEvt > nEvents) lastEvt = nEvents;
   if (firstEvt < 0) firstEvt = 0;
   nEvents = lastEvt-firstEvt;

   // use timer
   Timer timer( nEvents, GetName(), kTRUE );

   if (logProgress)
      Log() << kHEADER << Form("[%s] : ",DataInfo().GetName())
            << "Evaluation of " << GetMethodName() << " on "
            << (Data()->GetCurrentType() == Types::kTraining ? "training" : "testing")
            << " sample (" << nEvents << " events)" << Endl;

   float* data = new float[nEvents*fNVars];
   for (UInt_t i=0; i<nEvents; i++) {
      Data()->SetCurrentEvent(i);
      const TMVA::Event *e = GetEvent();
      for (UInt_t j=0; j<fNVars; j++) {
         data[j + i*fNVars] = e->GetValue(j);
      }
   }

   npy_intp dimsData[2] = {(npy_intp)nEvents, (npy_intp)fNVars};
   PyArrayObject* pDataMvaValues = (PyArrayObject*)PyArray_SimpleNewFromData(2, dimsData, NPY_FLOAT, (void*)data);
   if (pDataMvaValues==0) Log() << "Failed to load data to Python array" << Endl;


   // Get prediction for all events
   PyObject* pModel = PyDict_GetItemString(fLocalNS, "model");
   if (pModel==0) Log() << kFATAL << "Failed to get model Python object" << Endl;


   /////////////////////
   // TODO: PyTorch doesn't have a method to use direct predictions.
   //       Use external function predict for this functionality.
   /////////////////////

   PyArrayObject* pPredictions = (PyArrayObject*) PyObject_CallMethod(pModel, (char*)"predict", (char*)"O", pDataMvaValues);
   

   // Note: This will surely fail for now.
   if (pPredictions==0) Log() << kFATAL << "Predictions for PyTorch Not implemented yet" << Endl;
   delete[] data;


   // Load predictions to double vector
   // NOTE: The signal probability is given at the output
   std::vector<double> mvaValues(nEvents);
   float* predictionsData = (float*) PyArray_DATA(pPredictions);
   for (UInt_t i=0; i<nEvents; i++) {
      mvaValues[i] = (double) predictionsData[i*fNOutputs + TMVA::Types::kSignal];
   }

   if (logProgress) {
      Log() << kINFO
            << "Elapsed time for evaluation of " << nEvents <<  " events: "
            << timer.GetElapsedTime() << "       " << Endl;
   }

   return mvaValues;
}

std::vector<Float_t>& MethodPyTorch::GetRegressionValues() {
   // Check whether the model is setup
   // NOTE: unfortunately this is needed because during evaluation ProcessOptions is not called again
   if (!fModelIsSetup){
      // Setup the model and load weights
      SetupPyTorchModel(true);
   }

   // Get regression values
   const TMVA::Event* e = GetEvent();
   for (UInt_t i=0; i<fNVars; i++) fVals[i] = e->GetValue(i);

   /////////////////////
   // TODO: PyTorch doesn't have a method to use direct predictions.
   //       Convert or omit to get the Keras like predictions setup.
   /////////////////////


   // PyRunString("for i,p in enumerate(model.predict(vals)): output[i]=p\n",
   //             "Failed to get predictions");


   // Use inverse transformation of targets to get final regression values
   Event * eTrans = new Event(*e);
   for (UInt_t i=0; i<fNOutputs; ++i) {
      eTrans->SetTarget(i,fOutput[i]);
   }

   const Event* eTrans2 = GetTransformationHandler().InverseTransform(eTrans);
   for (UInt_t i=0; i<fNOutputs; ++i) {
      fOutput[i] = eTrans2->GetTarget(i);
   }

   return fOutput;
}

std::vector<Float_t>& MethodPyTorch::GetMulticlassValues() {
   // Check whether the model is setup
   // NOTE: unfortunately this is needed because during evaluation ProcessOptions is not called again
   if (!fModelIsSetup){
      // Setup the model and load weights
      SetupPyTorchModel(true);
   }

   // Get class probabilites
   const TMVA::Event* e = GetEvent();
   for (UInt_t i=0; i<fNVars; i++) fVals[i] = e->GetValue(i);

   /////////////////////
   // TODO: PyTorch doesn't have a method to use direct predictions.
   //       Convert or omit to get the Keras like predictions setup.
   /////////////////////


   PyRunString("for i,p in enumerate(model.predict(vals)): output[i]=p\n",
               "Failed to get predictions");


   return fOutput;
}

void MethodPyTorch::ReadModelFromFile() {
}

void MethodPyTorch::GetHelpMessage() const {
   Log() << Endl;
   Log() << "PyTorch is a scientific computing package supporting" << Endl;
   Log() << "automatic differentiation. This method wraps the training" << Endl;
   Log() << "and predictions steps of the PyTorch Python package for" << Endl;
   Log() << "TMVA, so that dataloading, preprocessing and evaluation" << Endl;
   Log() << "can be done within the TMVA system. To use this PyTorch" << Endl;
   Log() << "interface, you need to generatea model with PyTorch first." << Endl;
   Log() << "Then, this model can be loaded and trained in TMVA." << Endl;
   Log() << Endl;
}
