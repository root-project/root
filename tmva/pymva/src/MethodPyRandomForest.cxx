// @(#)root/tmva/pymva $Id$
// Authors: Omar Zapata, Lorenzo Moneta, Sergei Gleyzer 2015

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodPyRandomForest                                                  *
 * Web    : http://oproject.org                                                   *
 *                                                                                *
 * Description:                                                                   *
 *      Random Forest Classifiear from Scikit learn                               *
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
#include "TMVA/MethodPyRandomForest.h"
#include "TMVA/Tools.h"
#include "TMVA/Ranking.h"
#include "TMVA/Types.h"
#include "TMVA/Config.h"
#include "TMVA/PDF.h"
#include "TMVA/ClassifierFactory.h"

#include "TMVA/Results.h"



using namespace TMVA;

REGISTER_METHOD(PyRandomForest)

ClassImp(MethodPyRandomForest)

//_______________________________________________________________________
MethodPyRandomForest::MethodPyRandomForest(const TString &jobName,
      const TString &methodTitle,
      DataSetInfo &dsi,
      const TString &theOption,
      TDirectory *theTargetDir) :
   PyMethodBase(jobName, Types::kPyRandomForest, methodTitle, dsi, theOption, theTargetDir),
   n_estimators(10),
   criterion("gini"),
   max_depth("None"),
   min_samples_split(2),
   min_samples_leaf(1),
   min_weight_fraction_leaf(0),
   max_features("'auto'"),
   max_leaf_nodes("None"),
   bootstrap(kTRUE),
   oob_score(kFALSE),
   n_jobs(1),
   random_state("None"),
   verbose(0),
   warm_start(kFALSE),
   class_weight("None")
{
   // standard constructor for the PyRandomForest
   SetWeightFileDir(gConfig().GetIONames().fWeightFileDir);

}

//_______________________________________________________________________
MethodPyRandomForest::MethodPyRandomForest(DataSetInfo &theData, const TString &theWeightFile, TDirectory *theTargetDir)
   : PyMethodBase(Types::kPyRandomForest, theData, theWeightFile, theTargetDir),
     n_estimators(10),
     criterion("gini"),
     max_depth("None"),
     min_samples_split(2),
     min_samples_leaf(1),
     min_weight_fraction_leaf(0),
     max_features("'auto'"),
     max_leaf_nodes("None"),
     bootstrap(kTRUE),
     oob_score(kFALSE),
     n_jobs(1),
     random_state("None"),
     verbose(0),
     warm_start(kFALSE),
     class_weight("None")
{
   SetWeightFileDir(gConfig().GetIONames().fWeightFileDir);
}


//_______________________________________________________________________
MethodPyRandomForest::~MethodPyRandomForest(void)
{
}

//_______________________________________________________________________
Bool_t MethodPyRandomForest::HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets)
{
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   return kFALSE;
}


//_______________________________________________________________________
void MethodPyRandomForest::DeclareOptions()
{
   MethodBase::DeclareCompatibilityOptions();

   DeclareOptionRef(n_estimators, "NEstimators", "Integer, optional (default=10). The number of trees in the forest.");
   DeclareOptionRef(criterion, "Criterion", "//string, optional (default='gini') \
    The function to measure the quality of a split. Supported criteria are \
    'gini' for the Gini impurity and 'entropy' for the information gain. \
    Note: this parameter is tree-specific.");

   DeclareOptionRef(max_depth, "MaxDepth", "integer or None, optional (default=None) \
                                             The maximum depth of the tree. If None, then nodes are expanded until \
                                             all leaves are pure or until all leaves contain less than \
                                             min_samples_split samples. \
                                             Ignored if ``max_leaf_nodes`` is not None.");
   DeclareOptionRef(min_samples_split, "MinSamplesSplit", "integer, optional (default=2)\
    The minimum number of samples required to split an internal node.");

   DeclareOptionRef(min_samples_leaf, "MinSamplesLeaf", "integer, optional (default=1) \
    The minimum number of samples in newly created leaves.  A split is \
    discarded if after the split, one of the leaves would contain less then \
    ``min_samples_leaf`` samples.");
   DeclareOptionRef(min_weight_fraction_leaf, "MinWeightFractionLeaf", "//float, optional (default=0.) \
    The minimum weighted fraction of the input samples required to be at a \
    leaf node.");
   DeclareOptionRef(max_features, "MaxFeatures", "The number of features to consider when looking for the best split");
   DeclareOptionRef(max_leaf_nodes, "MaxLeafNodes", "int or None, optional (default=None)\
    Grow trees with ``max_leaf_nodes`` in best-first fashion.\
    Best nodes are defined as relative reduction in impurity.\
    If None then unlimited number of leaf nodes.\
    If not None then ``max_depth`` will be ignored.");
   DeclareOptionRef(bootstrap, "Bootstrap", "boolean, optional (default=True) \
    Whether bootstrap samples are used when building trees.");
   DeclareOptionRef(oob_score, "OoBScore", " bool Whether to use out-of-bag samples to estimate\
    the generalization error.");
   DeclareOptionRef(n_jobs, "NJobs", " integer, optional (default=1) \
    The number of jobs to run in parallel for both `fit` and `predict`. \
    If -1, then the number of jobs is set to the number of cores.");

   DeclareOptionRef(random_state, "RandomState", "int, RandomState instance or None, optional (default=None)\
    If int, random_state is the seed used by the random number generator;\
    If RandomState instance, random_state is the random number generator;\
    If None, the random number generator is the RandomState instance used\
    by `np.random`.");
   DeclareOptionRef(verbose, "Verbose", "int, optional (default=0)\
    Controls the verbosity of the tree building process.");
   DeclareOptionRef(warm_start, "WarmStart", "bool, optional (default=False)\
    When set to ``True``, reuse the solution of the previous call to fit\
    and add more estimators to the ensemble, otherwise, just fit a whole\
    new forest.");
   DeclareOptionRef(class_weight, "ClassWeight", "dict, list of dicts, \"auto\", \"subsample\" or None, optional\
    Weights associated with classes in the form ``{class_label: weight}``.\
    If not given, all classes are supposed to have weight one. For\
    multi-output problems, a list of dicts can be provided in the same\
    order as the columns of y.\
    The \"auto\" mode uses the values of y to automatically adjust\
    weights inversely proportional to class frequencies in the input data.\
    The \"subsample\" mode is the same as \"auto\" except that weights are\
    computed based on the bootstrap sample for every tree grown.\
    For multi-output, the weights of each column of y will be multiplied.\
    Note that these weights will be multiplied with sample_weight (passed\
    through the fit method) if sample_weight is specified.");
}

//_______________________________________________________________________
void MethodPyRandomForest::ProcessOptions()
{
   if (n_estimators <= 0) {
      Log() << kERROR << " NEstimators <=0... that does not work !! "
            << " I set it to 10 .. just so that the program does not crash"
            << Endl;
      n_estimators = 10;
   }
   if (criterion != "gini" && criterion != "entropy") {
      Log() << kFATAL << Form(" Criterion = %s... that does not work !! ", criterion.Data())
            << " The options are gini of entropy."
            << Endl;
   }
   PyObject *pomax_depth = Eval(max_depth);
   if (!pomax_depth) {
      Log() << kFATAL << Form(" MaxDepth = %s... that does not work !! ", criterion.Data())
            << " The options are None or integer."
            << Endl;
   }
   Py_DECREF(pomax_depth);

   if (min_samples_split < 0) {
      Log() << kERROR << " MinSamplesSplit < 0... that does not work !! "
            << " I set it to 2 .. just so that the program does not crash"
            << Endl;
      min_samples_split = 2;
   }
   if (min_samples_leaf < 0) {
      Log() << kERROR << " MinSamplesLeaf < 0... that does not work !! "
            << " I set it to 1 .. just so that the program does not crash"
            << Endl;
      min_samples_leaf = 1;
   }

   if (min_weight_fraction_leaf < 0) {
      Log() << kERROR << " MinWeightFractionLeaf < 0... that does not work !! "
            << " I set it to 0 .. just so that the program does not crash"
            << Endl;
      min_weight_fraction_leaf = 0;
   }
   if (max_features == "auto" || max_features == "sqrt" || max_features == "log2")max_features = Form("'%s'", max_features.Data());
   PyObject *pomax_features = Eval(max_features);
   if (!pomax_features) {
      Log() << kFATAL << Form(" MaxFeatures = %s... that does not work !! ", max_features.Data())
            << "int, float, string or None, optional (default='auto')"
            << "The number of features to consider when looking for the best split:"
            << "If int, then consider `max_features` features at each split."
            << "If float, then `max_features` is a percentage and"
            << "`int(max_features * n_features)` features are considered at each split."
            << "If 'auto', then `max_features=sqrt(n_features)`."
            << "If 'sqrt', then `max_features=sqrt(n_features)`."
            << "If 'log2', then `max_features=log2(n_features)`."
            << "If None, then `max_features=n_features`."
            << Endl;
   }
   Py_DECREF(pomax_features);

   PyObject *pomax_leaf_nodes = Eval(max_leaf_nodes);
   if (!pomax_leaf_nodes) {
      Log() << kFATAL << Form(" MaxLeafNodes = %s... that does not work !! ", max_leaf_nodes.Data())
            << " The options are None or integer."
            << Endl;
   }
   Py_DECREF(pomax_leaf_nodes);

//    bootstrap(kTRUE),
//    oob_score(kFALSE),
//    n_jobs(1),

   PyObject *porandom_state = Eval(random_state);
   if (!porandom_state) {
      Log() << kFATAL << Form(" RandomState = %s... that does not work !! ", random_state.Data())
            << "If int, random_state is the seed used by the random number generator;"
            << "If RandomState instance, random_state is the random number generator;"
            << "If None, the random number generator is the RandomState instance used by `np.random`."
            << Endl;
   }
   Py_DECREF(porandom_state);

//    verbose(0),
//    warm_start(kFALSE),
//    class_weight("None")
   PyObject *poclass_weight = Eval(class_weight);
   if (!poclass_weight) {
      Log() << kFATAL << Form(" ClassWeight = %s... that does not work !! ", class_weight.Data())
            << "dict, list of dicts, 'auto', 'subsample' or None, optional"
            << Endl;
   }
   Py_DECREF(poclass_weight);

}

//_______________________________________________________________________
void  MethodPyRandomForest::Init()
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

//_______________________________________________________________________
void MethodPyRandomForest::Train()
{
//        n_estimators(10),
//    criterion("gini"),
//    max_depth("None"),
//    min_samples_split(2),
//    min_samples_leaf(1),
//    min_weight_fraction_leaf(0.0),
//    max_features("'auto'"),
//    max_leaf_nodes("None"),
//    bootstrap(kTRUE),
//    oob_score(kFALSE),
//    n_jobs(1),
//    random_state("None"),
//    verbose(0),
//    warm_start(kFALSE),
//    class_weight("None")

   //NOTE: max_features must have 3 defferents variables int, float and string
   if (max_features == "auto" || max_features == "sqrt" || max_features == "log2")max_features = Form("'%s'", max_features.Data());
   PyObject *pomax_features = Eval(max_features);
   PyObject *pomax_depth = Eval(max_depth);
   PyObject *pomax_leaf_nodes = Eval(max_leaf_nodes);
   PyObject *porandom_state = Eval(random_state);
   PyObject *poclass_weight = Eval(class_weight);
//     PyObject_Print(pomax_features,stdout,0);
//     std::cout<<std::endl;
//
//     PyObject_Print(pomax_depth,stdout,0);
//     std::cout<<std::endl;
   PyObject *args = Py_BuildValue("(isOiifOOiiiOiiO)", n_estimators, criterion.Data(), pomax_depth, min_samples_split, \
                                  min_samples_leaf, min_weight_fraction_leaf, pomax_features, pomax_leaf_nodes, \
                                  bootstrap, oob_score, n_jobs, porandom_state, verbose, warm_start, poclass_weight);
   Py_DECREF(pomax_depth);
   PyObject_Print(args, stdout, 0);
   std::cout << std::endl;

   PyObject *pDict = PyModule_GetDict(fModule);
   PyObject *fClassifierClass = PyDict_GetItemString(pDict, "RandomForestClassifier");
   //    Log() << kFATAL <<"Train =" <<n_jobs<<Endl;

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
      Log() << kFATAL << "Can't call function RandomForestClassifier" << Endl;
      Log() << Endl;

   }

   fClassifier = PyObject_CallMethod(fClassifier, const_cast<char *>("fit"), const_cast<char *>("(OOO)"), fTrainData, fTrainDataClasses, fTrainDataWeights);
   //     PyObject_Print(fClassifier, stdout, 0);
   //     pValue =PyObject_CallObject(fClassifier, PyString_FromString("classes_"));
   //     PyObject_Print(pValue, stdout, 0);

   TString path = GetWeightFileDir() + "/PyRFModel.PyData";
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
void MethodPyRandomForest::TestClassification()
{
   MethodBase::TestClassification();
}


//_______________________________________________________________________
Double_t MethodPyRandomForest::GetMvaValue(Double_t *errLower, Double_t *errUpper)
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
   PyArrayObject *result = (PyArrayObject *)PyObject_CallMethod(fClassifier, const_cast<char *>("predict_proba"), const_cast<char *>("(O)"), pEvent);
   double *proba = (double *)(PyArray_DATA(result));
   mvaValue = proba[1]; //getting signal prob
   Py_DECREF(result);
   Py_DECREF(pEvent);
   return mvaValue;
}

//_______________________________________________________________________
void MethodPyRandomForest::ReadStateFromFile()
{
   if (!PyIsInitialized()) {
      PyInitialize();
   }

   TString path = GetWeightFileDir() + "/PyRFModel.PyData";
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
void MethodPyRandomForest::GetHelpMessage() const
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

