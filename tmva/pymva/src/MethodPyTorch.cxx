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

   // TODO: Check if verbosity is required in pytorch models.
   fVerbose = 1;
   
   fContinueTraining = false;
   fSaveBestOnly = true;
   fTriesEarlyStopping = -1;
   fLearningRateSchedule = ""; // empty string deactivates learning rate scheduler
   fFilenameTrainedModel = ""; // empty string sets output model filename to default (in weights/)
   fTensorBoard = "";          // empty string deactivates TensorBoard callback
}

MethodPyTorch::~MethodPyTorch() {
}

Bool_t MethodPyTorch::HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t) {
   if (type == Types::kRegression) return kTRUE;
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   if (type == Types::kMulticlass && numberClasses >= 2) return kTRUE;
   return kFALSE;
}