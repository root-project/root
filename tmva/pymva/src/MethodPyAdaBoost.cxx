// @(#)root/tmva/pymva $Id$
// Authors: Omar Zapata, Lorenzo Moneta, Sergei Gleyzer 2015

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodPyAdaBoost                                                  *
 * Web    : http://oproject.org                                           *
 *                                                                                *
 * Description:                                                                   *
 *      AdaBoost      Classifiear from Scikit learn                               *
 *                                                                                *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 *                                                                                *
 **********************************************************************************/
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <iomanip>
#include <fstream>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "TMath.h"
#include "Riostream.h"
#include "TMatrix.h"
#include "TMatrixD.h"
#include "TVectorD.h"

#include "TMVA/VariableTransformBase.h"
#include "TMVA/MethodPyAdaBoost.h"
#include "TMVA/Tools.h"
#include "TMVA/Ranking.h"
#include "TMVA/Types.h"
#include "TMVA/Config.h"
#include "TMVA/PDF.h"
#include "TMVA/ClassifierFactory.h"

#include "TMVA/Results.h"



using namespace TMVA;

REGISTER_METHOD(PyAdaBoost)

ClassImp(MethodPyAdaBoost)

//_______________________________________________________________________
MethodPyAdaBoost::MethodPyAdaBoost(const TString &jobName,
                                   const TString &methodTitle,
                                   DataSetInfo &dsi,
                                   const TString &theOption,
                                   TDirectory *theTargetDir) :
   PyMethodBase(jobName, Types::kPyAdaBoost, methodTitle, dsi, theOption, theTargetDir),
   base_estimator("None"),
   n_estimators(50),
   learning_rate(1.0),
   algorithm("SAMME.R"),
   random_state("None")
{
   // standard constructor for the PyAdaBoost
   SetWeightFileDir(gConfig().GetIONames().fWeightFileDir);

}

//_______________________________________________________________________
MethodPyAdaBoost::MethodPyAdaBoost(DataSetInfo &theData, const TString &theWeightFile, TDirectory *theTargetDir)
   : PyMethodBase(Types::kPyAdaBoost, theData, theWeightFile, theTargetDir),
     base_estimator("None"),
     n_estimators(50),
     learning_rate(1.0),
     algorithm("SAMME.R"),
     random_state("None")
{
   SetWeightFileDir(gConfig().GetIONames().fWeightFileDir);
}


//_______________________________________________________________________
MethodPyAdaBoost::~MethodPyAdaBoost(void)
{
}

//_______________________________________________________________________
Bool_t MethodPyAdaBoost::HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets)
{
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   return kFALSE;
}


//_______________________________________________________________________
void MethodPyAdaBoost::DeclareOptions()
{
   MethodBase::DeclareCompatibilityOptions();

   DeclareOptionRef(base_estimator, "BaseEstimator", "object, optional (default=DecisionTreeClassifier)\
    The base estimator from which the boosted ensemble is built.\
    Support for sample weighting is required, as well as proper `classes_`\
    and `n_classes_` attributes.");

   DeclareOptionRef(n_estimators, "NEstimators", "integer, optional (default=50)\
    The maximum number of estimators at which boosting is terminated.\
    In case of perfect fit, the learning procedure is stopped early.");

   DeclareOptionRef(learning_rate, "LearningRate", "float, optional (default=1.)\
    Learning rate shrinks the contribution of each classifier by\
    ``learning_rate``. There is a trade-off between ``learning_rate`` and\
    ``n_estimators``.");

   DeclareOptionRef(algorithm, "Algorithm", "{'SAMME', 'SAMME.R'}, optional (default='SAMME.R')\
    If 'SAMME.R' then use the SAMME.R real boosting algorithm.\
    ``base_estimator`` must support calculation of class probabilities.\
    If 'SAMME' then use the SAMME discrete boosting algorithm.\
    The SAMME.R algorithm typically converges faster than SAMME,\
    achieving a lower test error with fewer boosting iterations.");

   DeclareOptionRef(random_state, "RandomState", "int, RandomState instance or None, optional (default=None)\
    If int, random_state is the seed used by the random number generator;\
    If RandomState instance, random_state is the random number generator;\
    If None, the random number generator is the RandomState instance used\
    by `np.random`.");
}

//_______________________________________________________________________
void MethodPyAdaBoost::ProcessOptions()
{
   PyObject *pobase_estimator = Eval(base_estimator);
   if (!pobase_estimator) {
      Log() << kFATAL << Form(" BaseEstimator = %s... that does not work !! ", base_estimator.Data())
            << " The options are Object or None."
            << Endl;
   }
   Py_DECREF(pobase_estimator);

   if (n_estimators <= 0) {
      Log() << kERROR << " NEstimators <=0... that does not work !! "
            << " I set it to 10 .. just so that the program does not crash"
            << Endl;
      n_estimators = 10;
   }
   if (learning_rate <= 0) {
      Log() << kERROR << " LearningRate <=0... that does not work !! "
            << " I set it to 1.0 .. just so that the program does not crash"
            << Endl;
      learning_rate = 1.0;
   }

   if (algorithm != "SAMME" && algorithm != "SAMME.R") {
      Log() << kFATAL << Form(" Algorithm = %s... that does not work !! ", algorithm.Data())
            << " The options are SAMME of SAMME.R."
            << Endl;
   }
   PyObject *porandom_state = Eval(random_state);
   if (!porandom_state) {
      Log() << kFATAL << Form(" RandomState = %s... that does not work !! ", random_state.Data())
            << "If int, random_state is the seed used by the random number generator;"
            << "If RandomState instance, random_state is the random number generator;"
            << "If None, the random number generator is the RandomState instance used by `np.random`."
            << Endl;
   }
   Py_DECREF(porandom_state);
}


//_______________________________________________________________________
void  MethodPyAdaBoost::Init()
{
   ProcessOptions();
   _import_array();//require to use numpy arrays

   //Import sklearn
   // Convert the file name to a Python string.
   PyObject *pName = PyString_FromString("sklearn.ensemble");
   // Import the file as a Python module.
   fModule = PyImport_Import(pName);
   Py_DECREF(pName);

   if (!fModule) {
      Log() << kFATAL << "Can't import sklearn.ensemble" << Endl;
      Log() << Endl;
   }


   //Training data
   UInt_t fNvars = Data()->GetNVariables();
   int fNrowsTraining = Data()->GetNTrainingEvents(); //every row is an event, a class type and a weight
   int *dims = new int[2];
   dims[0] = fNrowsTraining;
   dims[1] = fNvars;
   fTrainData = (PyArrayObject *)PyArray_FromDims(2, dims, NPY_FLOAT);
   float *TrainData = (float *)(PyArray_DATA(fTrainData));


   fTrainDataClasses = (PyArrayObject *)PyArray_FromDims(1, &fNrowsTraining, NPY_FLOAT);
   float *TrainDataClasses = (float *)(PyArray_DATA(fTrainDataClasses));

   fTrainDataWeights = (PyArrayObject *)PyArray_FromDims(1, &fNrowsTraining, NPY_FLOAT);
   float *TrainDataWeights = (float *)(PyArray_DATA(fTrainDataWeights));

   for (int i = 0; i < fNrowsTraining; i++) {
      const TMVA::Event *e = Data()->GetTrainingEvent(i);
      for (UInt_t j = 0; j < fNvars; j++) {
         TrainData[j + i * fNvars] = e->GetValue(j);
      }
      if (e->GetClass() == TMVA::Types::kSignal) TrainDataClasses[i] = TMVA::Types::kSignal;
      else TrainDataClasses[i] = TMVA::Types::kBackground;

      TrainDataWeights[i] = e->GetWeight();
   }
}

void MethodPyAdaBoost::Train()
{
//    base_estimator("None"),
//    n_estimators(50),
//    learning_rate(1.0),
//    algorithm("SAMME.R"),
//    random_state("None")
   PyObject *pobase_estimator = Eval(base_estimator);
   PyObject *porandom_state = Eval(random_state);

   PyObject *args = Py_BuildValue("(OifsO)", pobase_estimator, n_estimators, learning_rate, algorithm.Data(), porandom_state);
   PyObject_Print(args, stdout, 0);
   std::cout << std::endl;
   PyObject *pDict = PyModule_GetDict(fModule);
   PyObject *fClassifierClass = PyDict_GetItemString(pDict, "AdaBoostClassifier");

   // Create an instance of the class
   if (PyCallable_Check(fClassifierClass)) {
      //instance
      fClassifier = PyObject_CallObject(fClassifierClass , args);
      PyObject_Print(fClassifier, stdout, 0);

      Py_DECREF(args);
   } else {
      PyErr_Print();
      Py_DECREF(pDict);
      Py_DECREF(fClassifierClass);
      Log() << kFATAL << "Can't call function AdaBoostClassifier" << Endl;
      Log() << Endl;

   }

   fClassifier = PyObject_CallMethod(fClassifier, (char *)"fit", (char *)"(OOO)", fTrainData, fTrainDataClasses, fTrainDataWeights);
//     PyObject_Print(fClassifier, stdout, 0);
//     std::cout<<std::endl;
   //     pValue =PyObject_CallObject(fClassifier, PyString_FromString("classes_"));
   //     PyObject_Print(pValue, stdout, 0);

   TString path = GetWeightFileDir() + "/PyAdaBoostModel.PyData";
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Saving State File In:" << gTools().Color("reset") << path << Endl;
   Log() << Endl;

   PyObject *model_arg = Py_BuildValue("(O)", fClassifier);
   PyObject *model_data = PyObject_CallObject(fPickleDumps , model_arg);
   std::ofstream PyData;
   PyData.open(path.Data());
   PyData << PyString_AsString(model_data);
   PyData.close();
   Py_DECREF(model_arg);
   Py_DECREF(model_data);
}

//_______________________________________________________________________
void MethodPyAdaBoost::TestClassification()
{
   MethodBase::TestClassification();
}


//_______________________________________________________________________
Double_t MethodPyAdaBoost::GetMvaValue(Double_t *errLower, Double_t *errUpper)
{
   // cannot determine error
   NoErrorCalc(errLower, errUpper);

   if (!fClassifier) ReadStateFromFile();

   Double_t mvaValue;
   const TMVA::Event *e = Data()->GetEvent();
   UInt_t nvars = e->GetNVariables();
   PyObject *pEvent = PyTuple_New(nvars);
   for (UInt_t i = 0; i < nvars; i++) {

      PyObject *pValue = PyFloat_FromDouble(e->GetValue(i));
      if (!pValue) {
         Py_DECREF(pEvent);
         Py_DECREF(fTrainData);
         Log() << kFATAL << "Error Evaluating MVA " << Endl;
      }
      PyTuple_SetItem(pEvent, i, pValue);
   }
   PyArrayObject *result = (PyArrayObject *)PyObject_CallMethod(fClassifier, (char *)"predict_proba", (char *)"(O)", pEvent);
   double *proba = (double *)(PyArray_DATA(result));
   mvaValue = proba[1]; //getting signal prob
   Py_DECREF(result);
   Py_DECREF(pEvent);
   return mvaValue;
}

//_______________________________________________________________________
void MethodPyAdaBoost::ReadStateFromFile()
{
   if (!PyIsInitialized()) {
      PyInitialize();
   }

   TString path = GetWeightFileDir() + "/PyAdaBoostModel.PyData";
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Loading State File From:" << gTools().Color("reset") << path << Endl;
   Log() << Endl;
   std::ifstream PyData;
   std::stringstream PyDataStream;
   std::string PyDataString;

   PyData.open(path.Data());
   PyDataStream << PyData.rdbuf();
   PyDataString = PyDataStream.str();
   PyData.close();

//   std::cout<<"-----------------------------------\n";
//   std::cout<<PyDataString.c_str();
//   std::cout<<"-----------------------------------\n";
   PyObject *model_arg = Py_BuildValue("(s)", PyDataString.c_str());
   fClassifier = PyObject_CallObject(fPickleLoads , model_arg);


   Py_DECREF(model_arg);
}

//_______________________________________________________________________
void MethodPyAdaBoost::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line:
   //         "|--------------------------------------------------------------|"
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Short description:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "Decision Trees and Rule-Based Models " << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance optimisation:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance tuning via configuration options:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << "<None>" << Endl;
}

