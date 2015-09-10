// @(#)root/tmva/pymva $Id$
// Authors: Omar Zapata, Lorenzo Moneta, Sergei Gleyzer 2015

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodPyGTB                                                           *
 * Web    : http://oproject.org                                                   *
 *                                                                                *
 * Description:                                                                   *
 *      GradientBoostingClassifier Classifiear from Scikit learn                  *
 *                                                                                *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 *                                                                                *
 **********************************************************************************/

#include <iomanip>
#include <fstream>

#include "TMath.h"
#include "Riostream.h"
#include "TMatrix.h"
#include "TMatrixD.h"
#include "TVectorD.h"

#include "TMVA/VariableTransformBase.h"
#include "TMVA/MethodPyGTB.h"
#include "TMVA/Tools.h"
#include "TMVA/Ranking.h"
#include "TMVA/Types.h"
#include "TMVA/Config.h"
#include "TMVA/PDF.h"
#include "TMVA/ClassifierFactory.h"

#include "TMVA/Results.h"



using namespace TMVA;

REGISTER_METHOD(PyGTB)

ClassImp(MethodPyGTB)

//_______________________________________________________________________
MethodPyGTB::MethodPyGTB(const TString &jobName,
                         const TString &methodTitle,
                         DataSetInfo &dsi,
                         const TString &theOption,
                         TDirectory *theTargetDir) :
   PyMethodBase(jobName, Types::kPyGTB, methodTitle, dsi, theOption, theTargetDir),
   loss("deviance"),
   learning_rate(0.1),
   n_estimators(100),
   subsample(1.0),
   min_samples_split(2),
   min_samples_leaf(1),
   min_weight_fraction_leaf(0.0),
   max_depth(3),
   init("None"),
   random_state("None"),
   max_features("None"),
   verbose(0),
   max_leaf_nodes("None"),
   warm_start(kFALSE)
{
   // standard constructor for the PyGTB
   SetWeightFileDir(gConfig().GetIONames().fWeightFileDir);

}

//_______________________________________________________________________
MethodPyGTB::MethodPyGTB(DataSetInfo &theData, const TString &theWeightFile, TDirectory *theTargetDir)
   : PyMethodBase(Types::kPyGTB, theData, theWeightFile, theTargetDir),
     loss("deviance"),
     learning_rate(0.1),
     n_estimators(100),
     subsample(1.0),
     min_samples_split(2),
     min_samples_leaf(1),
     min_weight_fraction_leaf(0.0),
     max_depth(3),
     init("None"),
     random_state("None"),
     max_features("None"),
     verbose(0),
     max_leaf_nodes("None"),
     warm_start(kFALSE)
{
   SetWeightFileDir(gConfig().GetIONames().fWeightFileDir);
}


//_______________________________________________________________________
MethodPyGTB::~MethodPyGTB(void)
{
}

//_______________________________________________________________________
Bool_t MethodPyGTB::HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets)
{
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   return kFALSE;
}


//_______________________________________________________________________
void MethodPyGTB::DeclareOptions()
{
   MethodBase::DeclareCompatibilityOptions();

   DeclareOptionRef(loss, "Loss", "{'deviance', 'exponential'}, optional (default='deviance')\
    loss function to be optimized. 'deviance' refers to\
    deviance (= logistic regression) for classification\
    with probabilistic outputs. For loss 'exponential' gradient\
    boosting recovers the AdaBoost algorithm.");

   DeclareOptionRef(learning_rate, "LearningRate", "float, optional (default=0.1)\
    learning rate shrinks the contribution of each tree by `learning_rate`.\
    There is a trade-off between learning_rate and n_estimators.");

   DeclareOptionRef(n_estimators, "NEstimators", "int (default=100)\
    The number of boosting stages to perform. Gradient boosting\
    is fairly robust to over-fitting so a large number usually\
    results in better performance.");

   DeclareOptionRef(subsample, "Subsample", "float, optional (default=1.0)\
    The fraction of samples to be used for fitting the individual base\
    learners. If smaller than 1.0 this results in Stochastic Gradient\
    Boosting. `subsample` interacts with the parameter `n_estimators`.\
    Choosing `subsample < 1.0` leads to a reduction of variance\
    and an increase in bias.");

   DeclareOptionRef(min_samples_split, "MinSamplesSplit", "integer, optional (default=2)\
    The minimum number of samples required to split an internal node.");

   DeclareOptionRef(min_samples_leaf, "MinSamplesLeaf", "integer, optional (default=1) \
    The minimum number of samples in newly created leaves.  A split is \
    discarded if after the split, one of the leaves would contain less then \
    ``min_samples_leaf`` samples.");

   DeclareOptionRef(min_weight_fraction_leaf, "MinWeightFractionLeaf", "//float, optional (default=0.) \
    The minimum weighted fraction of the input samples required to be at a \
    leaf node.");

   DeclareOptionRef(max_depth, "MaxDepth", "integer or None, optional (default=None) \
                                             The maximum depth of the tree. If None, then nodes are expanded until \
                                             all leaves are pure or until all leaves contain less than \
                                             min_samples_split samples. \
                                             Ignored if ``max_leaf_nodes`` is not None.");

   DeclareOptionRef(init, "Init", "BaseEstimator, None, optional (default=None)\
    An estimator object that is used to compute the initial\
    predictions. ``init`` has to provide ``fit`` and ``predict``.\
    If None it uses ``loss.init_estimator`");

   DeclareOptionRef(random_state, "RandomState", "int, RandomState instance or None, optional (default=None)\
    If int, random_state is the seed used by the random number generator;\
    If RandomState instance, random_state is the random number generator;\
    If None, the random number generator is the RandomState instance used\
    by `np.random`.");
   DeclareOptionRef(max_features, "MaxFeatures", "The number of features to consider when looking for the best split");
   DeclareOptionRef(verbose, "Verbose", "int, optional (default=0)\
    Controls the verbosity of the tree building process.");
   DeclareOptionRef(max_leaf_nodes, "MaxLeafNodes", "int or None, optional (default=None)\
    Grow trees with ``max_leaf_nodes`` in best-first fashion.\
    Best nodes are defined as relative reduction in impurity.\
    If None then unlimited number of leaf nodes.\
    If not None then ``max_depth`` will be ignored.");
   DeclareOptionRef(warm_start, "WarmStart", "bool, optional (default=False)\
    When set to ``True``, reuse the solution of the previous call to fit\
    and add more estimators to the ensemble, otherwise, just fit a whole\
    new forest.");




}

//_______________________________________________________________________
void MethodPyGTB::ProcessOptions()
{
   if (loss != "deviance" && loss != "exponential") {
      Log() << kFATAL << Form(" Loss = %s... that does not work !! ", loss.Data())
            << " The options are deviance of exponential."
            << Endl;
   }

   if (learning_rate <= 0) {
      Log() << kERROR << " LearningRate <=0... that does not work !! "
            << " I set it to 0.1 .. just so that the program does not crash"
            << Endl;
      learning_rate = 0.1;
   }
   if (n_estimators <= 0) {
      Log() << kERROR << " NEstimators <=0... that does not work !! "
            << " I set it to 100 .. just so that the program does not crash"
            << Endl;
      n_estimators = 100;
   }
   if (min_samples_split < 0) {
      Log() << kERROR << " MinSamplesSplit <0... that does not work !! "
            << " I set it to 2 .. just so that the program does not crash"
            << Endl;
      min_samples_split = 2;
   }
   if (subsample < 0) {
      Log() << kERROR << " Subsample <0... that does not work !! "
            << " I set it to 1.0 .. just so that the program does not crash"
            << Endl;
      subsample = 1.0;
   }

   if (min_samples_leaf < 0) {
      Log() << kERROR << " MinSamplesLeaf <0... that does not work !! "
            << " I set it to 1.0 .. just so that the program does not crash"
            << Endl;
      min_samples_leaf = 1;
   }

   if (min_samples_leaf < 0) {
      Log() << kERROR << " MinSamplesLeaf <0... that does not work !! "
            << " I set it to 1.0 .. just so that the program does not crash"
            << Endl;
      min_samples_leaf = 1;
   }

   if (min_weight_fraction_leaf < 0) {
      Log() << kERROR << " MinWeightFractionLeaf <0... that does not work !! "
            << " I set it to 0.0 .. just so that the program does not crash"
            << Endl;
      min_weight_fraction_leaf = 0.0;
   }

   if (max_depth < 0) {
      Log() << kERROR << " MaxDepth <0... that does not work !! "
            << " I set it to 3 .. just so that the program does not crash"
            << Endl;
      max_depth = 3;
   }

   PyObject *poinit = Eval(init);
   if (!poinit) {
      Log() << kFATAL << Form(" Init = %s... that does not work !! ", init.Data())
            << " The options are None or  BaseEstimator. An estimator object that is used to compute the initial"
            << " predictions. ``init`` has to provide ``fit`` and ``predict``."
            << " If None it uses ``loss.init_estimator``."
            << Endl;
   }
   Py_DECREF(poinit);

   PyObject *porandom_state = Eval(random_state);
   if (!porandom_state) {
      Log() << kFATAL << Form(" RandomState = %s... that does not work !! ", random_state.Data())
            << "If int, random_state is the seed used by the random number generator;"
            << "If RandomState instance, random_state is the random number generator;"
            << "If None, the random number generator is the RandomState instance used by `np.random`."
            << Endl;
   }
   Py_DECREF(porandom_state);

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

//    verbose(0),
   PyObject *pomax_leaf_nodes = Eval(max_leaf_nodes);
   if (!pomax_leaf_nodes) {
      Log() << kFATAL << Form(" MaxLeafNodes = %s... that does not work !! ", max_leaf_nodes.Data())
            << " The options are None or integer."
            << Endl;
   }
   Py_DECREF(pomax_leaf_nodes);

//    warm_start(kFALSE)

}


//_______________________________________________________________________
void  MethodPyGTB::Init()
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

void MethodPyGTB::Train()
{
//    loss("deviance"),
//    learning_rate(0.1),
//    n_estimators(100),
//    subsample(1.0),
//    min_samples_split(2),
//    min_samples_leaf(1),
//    min_weight_fraction_leaf(0.0),
//    max_depth(3),
//    init("None"),
//    random_state("None"),
//    max_features("None"),
//    verbose(0),
//    max_leaf_nodes("None"),
//    warm_start(kFALSE)

   //NOTE: max_features must have 3 defferents variables int, float and string
   //search a solution with PyObject
   PyObject *poinit = Eval(init);
   PyObject *porandom_state = Eval(random_state);
   PyObject *pomax_features = Eval(max_features);
   PyObject *pomax_leaf_nodes = Eval(max_leaf_nodes);

   PyObject *args = Py_BuildValue("(sfifiifiOOOiOi)", loss.Data(), \
                                  learning_rate, n_estimators, subsample, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, \
                                  max_depth, poinit, porandom_state, pomax_features, verbose, pomax_leaf_nodes, warm_start);

   PyObject_Print(args, stdout, 0);
   std::cout << std::endl;

   PyObject *pDict = PyModule_GetDict(fModule);
   PyObject *fClassifierClass = PyDict_GetItemString(pDict, "GradientBoostingClassifier");

   // Create an instance of the class
   if (PyCallable_Check(fClassifierClass)) {
      //instance
      fClassifier = PyObject_CallObject(fClassifierClass , args);
      PyObject_Print(fClassifier, stdout, 0);
      std::cout << std::endl;

      Py_DECREF(poinit);
      Py_DECREF(porandom_state);
      Py_DECREF(pomax_features);
      Py_DECREF(pomax_leaf_nodes);
      Py_DECREF(args);
   } else {
      PyErr_Print();
      Py_DECREF(poinit);
      Py_DECREF(porandom_state);
      Py_DECREF(pomax_features);
      Py_DECREF(pomax_leaf_nodes);
      Py_DECREF(args);
      Py_DECREF(pDict);
      Py_DECREF(fClassifierClass);
      Log() << kFATAL << "Can't call function GradientBoostingClassifier" << Endl;
      Log() << Endl;

   }

   fClassifier = PyObject_CallMethod(fClassifier, (char *)"fit", (char *)"(OOO)", fTrainData, fTrainDataClasses, fTrainDataWeights);
//     PyObject_Print(fClassifier, stdout, 0);
//     std::cout<<std::endl;
   //     pValue =PyObject_CallObject(fClassifier, PyString_FromString("classes_"));
   //     PyObject_Print(pValue, stdout, 0);

   TString path = GetWeightFileDir() + "/PyGTBModel.PyData";
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
void MethodPyGTB::TestClassification()
{
   MethodBase::TestClassification();
}


//_______________________________________________________________________
Double_t MethodPyGTB::GetMvaValue(Double_t *errLower, Double_t *errUpper)
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
void MethodPyGTB::ReadStateFromFile()
{
   if (!PyIsInitialized()) {
      PyInitialize();
   }

   TString path = GetWeightFileDir() + "/PyGTBModel.PyData";
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
void MethodPyGTB::GetHelpMessage() const
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

