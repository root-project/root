// @(#)root/tmva/pymva $Id$
// Author: Anirudh Dagar, 2020

#include <Python.h>
#include "TMVA/MethodPyTorch.h"

// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// #include <numpy/arrayobject.h>

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
   SetupTorchModel(fContinueTraining);
}


void MethodPyTorch::SetupTorchModel(bool loadTrainedModel) {
   /*
    * Load PyTorch model from file
    */

   Log() << kINFO << " Setup Torch Model " << Endl;

   PyRunString("load_model_custom_objects=None");


   if (!fUserCodeName.IsNull()) {
      Log() << kINFO << " Executing user initialization code from  " << fUserCodeName << Endl;


        // run some python code provided by user for model initialization if needed
      TString cmd = "exec(open('" + fUserCodeName + "').read())";
      TString errmsg = "Error executing the provided user code";
      PyRunString(cmd, errmsg);

      PyRunString("print('custom objects for loading model : ',load_model_custom_objects)");
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

   // Import torch
   PyRunString("import sys; sys.argv = ['']", "Set sys.argv failed");
   PyRunString("import torch", "Import torch failed");

   // Set flag that model is not setup
   fModelIsSetup = false;
}