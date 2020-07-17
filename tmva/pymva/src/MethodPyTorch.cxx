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