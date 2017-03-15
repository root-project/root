// @(#)root/tmva/pymva $Id$
// Authors: Omar Zapata, Lorenzo Moneta, Sergei Gleyzer 2015, Satyarth Praveen

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodPyKMeans                                                  *
 * Web    : http://oproject.org                                                   *
 *                                                                                *
 * Description:                                                                   *
 *      K-Means from Scikit learn                               *
 *                                                                                *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 *                                                                                *
 **********************************************************************************/
#include <Python.h>    // Needs to be included first to avoid redefinition of _POSIX_C_SOURCE
#include "TMVA/MethodPyKMeans.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "TMVA/Configurable.h"
#include "TMVA/ClassifierFactory.h"
#include "TMVA/Config.h"
#include "TMVA/DataSet.h"
#include "TMVA/Event.h"
#include "TMVA/IMethod.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/PDF.h"
#include "TMVA/Ranking.h"
#include "TMVA/Results.h"
#include "TMVA/Tools.h"
#include "TMVA/Types.h"
#include "TMVA/VariableTransformBase.h"

#include "Riostream.h"
#include "TMath.h"
#include "TMatrix.h"
#include "TMatrixD.h"
#include "TVectorD.h"

#include <iomanip>
#include <fstream>

using namespace TMVA;

REGISTER_METHOD(PyKMeans)

ClassImp(MethodPyKMeans)

//_______________________________________________________________________
MethodPyKMeans::MethodPyKMeans(const TString &jobName,
      const TString &methodTitle,
      DataSetInfo &dsi,
      const TString &theOption) :
   PyMethodBase(jobName, Types::kPyKMeans, methodTitle, dsi, theOption),
   n_clusters(8),
   init("'k-means++'"),
   n_init(10),
   max_iter(300),
   tol(0.0001),
   precompute_distances("'auto'"),
   verbose(0),
   random_state("None"),
   copy_x(kTRUE),
   n_jobs(1)
   // algorithm("auto"),    // This parameter is available in scikit-learn version 0.18
{
}

//_______________________________________________________________________
MethodPyKMeans::MethodPyKMeans(DataSetInfo &theData, const TString &theWeightFile)
   : PyMethodBase(Types::kPyKMeans, theData, theWeightFile),
     n_clusters(8),
     init("'k-means++'"),
     n_init(10),
     max_iter(300),
     tol(0.0001),
     precompute_distances("'auto'"),
     verbose(0),
     random_state("None"),
     copy_x(kTRUE),
     n_jobs(1)
     // algorithm("auto"),    // This parameter is available in scikit-learn version 0.18
{
}


//_______________________________________________________________________
MethodPyKMeans::~MethodPyKMeans(void)
{
}

//_______________________________________________________________________
Bool_t MethodPyKMeans::HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets)
{
   if (type == Types::kClassification && numberClasses == 2) return kTRUE;
   return kFALSE;
}


//_______________________________________________________________________
void MethodPyKMeans::DeclareOptions()
{
   MethodBase::DeclareCompatibilityOptions();

   DeclareOptionRef(n_clusters, "NClusters", "Integer, optional, default: 8. The number of clusters to form as well as the number of centroids to generate.");

   DeclareOptionRef(init, "Init", "{‘k-means++’, ‘random’ or an ndarray}. Method for initialization, defaults to ‘k-means++’: ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details. ‘random’: choose k observations (rows) at random from data for the initial centroids. If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.");
   
   DeclareOptionRef(n_init, "NInit", "Integer, default: 10. Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.");
   
   DeclareOptionRef(max_iter, "MaxIter", "Integer, default: 300. Maximum number of iterations of the k-means algorithm for a single run.");

   DeclareOptionRef(tol, "Tol", "float, default: 1e-4. Relative tolerance with regards to inertia to declare convergence");

   DeclareOptionRef(precompute_distances, "PrecomputeDistances", "{‘auto’, True, False}. Precompute distances (faster but takes more memory). ‘auto’ : do not precompute distances if n_samples * n_clusters > 12 million. This corresponds to about 100MB overhead per job using double precision. True : always precompute distances. False : never precompute distances");

   DeclareOptionRef(verbose, "Verbose", "int, default 0. Verbosity mode.");

   DeclareOptionRef(random_state, "RandomState", "integer or numpy.RandomState, optional. The generator used to initialize the centers. If an integer is given, it fixes the seed. Defaults to the global numpy random number generator.");

   DeclareOptionRef(copy_x, "CopyX", "boolean, default True. When pre-computing distances it is more numerically accurate to center the data first. If copy_x is True, then the original data is not modified. If False, the original data is modified, and put back before the function returns, but small numerical differences may be introduced by subtracting and then adding the data mean.");

   DeclareOptionRef(n_jobs, "NJobs", "Integer. The number of jobs to use for the computation. This works by computing each of the n_init runs in parallel. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.");

   // This parameter is available in scikit-learn version 0.18
   // DeclareOptionRef(algorithm, "Algorithm", "\"auto\", \"full\" or \"elkan\", default=\"auto\" K-means algorithm to use. The classical EM-style algorithm is \"full\". The \"elkan\" variation is more efficient by using the triangle inequality, but currently doesn’t support sparse data. \"auto\" chooses \"elkan\" for dense data and \"full\" for sparse data.");

}

//_______________________________________________________________________
void MethodPyKMeans::ProcessOptions()
{
  if (n_clusters <= 0) {
    Log() << kERROR << " NClusters <=0... that does not work !! "
          << " I set it to 8 .. just so that the program does not crash"
          << Endl;
    n_clusters = 8;
  }
   
  PyObject *poinit = Eval(init);
  if(!poinit) {
    Log() << kFATAL << Form(" Init = %s... that does not work !! ", init.Data())
          << "{‘k-means++’, ‘random’ or an ndarray}"
          << "Method for initialization, defaults to ‘k-means++’:"
          << "‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence."
          << "‘random’: choose k observations (rows) at random from data for the initial centroids."
          << "If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers."
          << Endl;
  }
  Py_DECREF(poinit);

  if (n_init <= 0) {
    Log() << kERROR << " NInit <=0... that does not work !! "
          << " I set it to 10 .. just so that the program does not crash"
          << Endl;
    n_init = 10;
  }

  if (max_iter <= 0) {
    Log() << kERROR << " MaxIter <=0... that does not work !! "
          << " I set it to 300 .. just so that the program does not crash"
          << Endl;
    max_iter = 300;
  }

  // tol(0.0001)
   
  PyObject *poprecompute_distances = Eval(precompute_distances);
  if(!poprecompute_distances) {
    Log() << kFATAL << Form(" PrecomputeDistances = %s... that does not work !! ", precompute_distances.Data())
          << "The options are 'auto', True, or False"
          << Endl;
  }
  Py_DECREF(poprecompute_distances);
   
  // verbose(0)

  // random_state(None)
  PyObject *porandom_state = Eval(random_state);
  if (!porandom_state) {
    Log() << kFATAL << Form(" RandomState = %s... that does not work !! ", random_state.Data())
          << "The generator used to initialize the centers."
          << "If an integer is given, it fixes the seed."
          << "Defaults to the global numpy random number generator."
          << Endl;
  }
  Py_DECREF(porandom_state);

  // copy_x(True)
  
  // n_jobs(1)

  // This parameter is available in scikit-learn version 0.18
  // if(algorithm!="auto" && algorithm!="full" && algorithm!="elkan"){
  //   Log() << kFATAL << Form(" Algorithm=%s... that does not work !! ", algorithm.Data())
  //         << " The options are \"auto\", \"full\", or \"elkan\""
  //         << Endl;
  // }

}




//_______________________________________________________________________
void  MethodPyKMeans::Init()
{
   ProcessOptions();
   _import_array();//require to use numpy arrays

   //Import sklearn
   // Convert the file name to a Python string.
   PyObject *pName = PyUnicode_FromString("sklearn.cluster");
   // Import the file as a Python module.
   fModule = PyImport_Import(pName);
   Py_DECREF(pName);

   if (!fModule) {
      Log() << kFATAL << "Can't import sklearn.cluster" << Endl;
      Log() << Endl;
   }


   //Training data
   UInt_t fNvars = Data()->GetNVariables();
   int fNrowsTraining = Data()->GetNTrainingEvents(); //every row is an event, a class type and a weight
   int dims[2];
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
void MethodPyKMeans::Train()
{

  PyObject *poinit = Eval(init);
  PyObject *poprecompute_distances = Eval(precompute_distances);
  PyObject *porandom_state = Eval(random_state);

  PyObject *args = Py_BuildValue("(iOiifOiOii)", n_clusters, poinit, n_init, max_iter, tol, poprecompute_distances, verbose, porandom_state, copy_x, n_jobs);
  // PyObject *args = Py_BuildValue("(iiiOOfiiii)", n_clusters, max_iter, n_init, poinit, poprecompute_distances, tol, n_jobs, random_state, verbose, copy_x);

  PyObject_Print(args, stdout, 0);
  std::cout << std::endl;

  PyObject *pDict = PyModule_GetDict(fModule);
  PyObject *fClassifierClass = PyDict_GetItemString(pDict, "KMeans");
  //    Log() << kFATAL <<"Train =" <<n_jobs<<Endl;

  // Create an instance of the class
  if (PyCallable_Check(fClassifierClass)) {
    //instance
    fClassifier = PyObject_CallObject(fClassifierClass , args);
    PyObject_Print(fClassifier, stdout, 0);
    std::cout << std::endl;

    Py_DECREF(args);
  } else {
    PyErr_Print();
    Py_DECREF(pDict);
    Py_DECREF(fClassifierClass);
    Log() << kFATAL << "Can't call function KMeans" << Endl;
    Log() << Endl;

  }
  
  // fClassifier = PyObject_CallMethod(fClassifier, const_cast<char *>("fit"), const_cast<char *>("(OOO)"), fTrainData, fTrainDataClasses, fTrainDataWeights);
  fClassifier = PyObject_CallMethod(fClassifier, const_cast<char *>("fit"), const_cast<char *>("(OO)"), fTrainData, fTrainDataClasses);

  if(!fClassifier)
  {
    Log() << kFATAL << "Can't create classifier object from KMeans" << Endl;
    Log() << Endl;  
  }
  if (IsModelPersistence())
  {
    TString path = GetWeightFileDir() + "/PyKMeansModel.PyData";
    Log() << Endl;
    Log() << gTools().Color("bold") << "--- Saving State File In:" << gTools().Color("reset") << path << Endl;
    Log() << Endl;
    Serialize(path,fClassifier);
  }
}

//_______________________________________________________________________
void MethodPyKMeans::TestClassification()
{
  MethodBase::TestClassification();
}


//_______________________________________________________________________
Double_t MethodPyKMeans::GetMvaValue(Double_t *errLower, Double_t *errUpper)
{
   // cannot determine error
   NoErrorCalc(errLower, errUpper);

   if (IsModelPersistence()) ReadModelFromFile();

   Double_t mvaValue;
   const TMVA::Event *e = Data()->GetEvent();
   UInt_t nvars = e->GetNVariables();
   int dims[2];
   dims[0] = 1;
   dims[1] = nvars;
   PyArrayObject *pEvent= (PyArrayObject *)PyArray_FromDims(2, dims, NPY_FLOAT);
   float *pValue = (float *)(PyArray_DATA(pEvent));

   for (UInt_t i = 0; i < nvars; i++) pValue[i] = e->GetValue(i);
   
   PyArrayObject *result = (PyArrayObject *)PyObject_CallMethod(fClassifier, const_cast<char *>("predict_proba"), const_cast<char *>("(O)"), pEvent);
   double *proba = (double *)(PyArray_DATA(result));
   mvaValue = proba[0]; //getting signal prob
   Py_DECREF(result);
   Py_DECREF(pEvent);
   return mvaValue;
}

//_______________________________________________________________________
void MethodPyKMeans::ReadModelFromFile()
{
   if (!PyIsInitialized()) {
      PyInitialize();
   }

   TString path = GetWeightFileDir() + "/PyKMeansModel.PyData";
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Loading State File From:" << gTools().Color("reset") << path << Endl;
   Log() << Endl;
   UnSerialize(path,&fClassifier);
   if(!fClassifier)
   {
     Log() << kFATAL << "Can't load KMeans from Serialized data." << Endl;
     Log() << Endl;
   }
}

//_______________________________________________________________________
void MethodPyKMeans::GetHelpMessage() const
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

