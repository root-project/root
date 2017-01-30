// @(#)root/tmva/pymva $Id$
// Authors: Omar Zapata, Lorenzo Moneta, Sergei Gleyzer 2015

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : PyMethodBase                                                          *
 *                                                                                *
 * Description:                                                                   *
 *      Virtual base class for all MVA method based on python                     *
 *                                                                                *
 **********************************************************************************/

#include <Python.h>    // Needs to be included first to avoid redefinition of _POSIX_C_SOURCE
#include <TMVA/PyMethodBase.h>

#include "TMVA/DataSet.h"
#include "TMVA/DataSetInfo.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/Results.h"
#include "TMVA/Timer.h"

#include <TApplication.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <fstream>
#include <wchar.h>

using namespace TMVA;

ClassImp(PyMethodBase)

// NOTE: Introduce here nothing that breaks if multiple instances
// of the same method share these objects, e.g., the local namespace.
PyObject *PyMethodBase::fModuleBuiltin = NULL;
PyObject *PyMethodBase::fEval = NULL;
PyObject *PyMethodBase::fOpen = NULL;

PyObject *PyMethodBase::fModulePickle = NULL;
PyObject *PyMethodBase::fPickleDumps = NULL;
PyObject *PyMethodBase::fPickleLoads = NULL;

PyObject *PyMethodBase::fMain = NULL;
PyObject *PyMethodBase::fGlobalNS = NULL;

class PyGILRAII {
   PyGILState_STATE m_GILState;
public:
   PyGILRAII():m_GILState(PyGILState_Ensure()){}
   ~PyGILRAII(){PyGILState_Release(m_GILState);}
};

//_______________________________________________________________________
PyMethodBase::PyMethodBase(const TString &jobName,
                           Types::EMVA methodType,
                           const TString &methodTitle,
                           DataSetInfo &dsi,
                           const TString &theOption ): MethodBase(jobName, methodType, methodTitle, dsi, theOption),
   fClassifier(NULL)
{
   if (!PyIsInitialized()) {
      PyInitialize();
   }

   // Set up private local namespace for each method instance
   fLocalNS = PyDict_New();
   if (!fLocalNS) {
      Log() << kFATAL << "Can't init local namespace" << Endl;
   }
}

//_______________________________________________________________________
PyMethodBase::PyMethodBase(Types::EMVA methodType,
                           DataSetInfo &dsi,
                           const TString &weightFile): MethodBase(methodType, dsi, weightFile),
   fClassifier(NULL)
{
   if (!PyIsInitialized()) {
      PyInitialize();
   }

   // Set up private local namespace for each method instance
   fLocalNS = PyDict_New();
   if (!fLocalNS) {
      Log() << kFATAL << "Can't init local namespace" << Endl;
   }
}

//_______________________________________________________________________
PyMethodBase::~PyMethodBase()
{
}

//_______________________________________________________________________
PyObject *PyMethodBase::Eval(TString code)
{
   if(!PyIsInitialized()) PyInitialize();
   PyObject *pycode = Py_BuildValue("(sOO)", code.Data(), fGlobalNS, fLocalNS);
   PyObject *result = PyObject_CallObject(fEval, pycode);
   Py_DECREF(pycode);
   return result;
}

//_______________________________________________________________________
// NOTE: We introduce a shared global namespace fGlobalNS, but using
// a private local namespace fLocalNS. This prohibits the interference
// of instances of the same method with the same factory, e.g., by overriding
// variables in the same local namespace.
void PyMethodBase::PyInitialize()
{
   TMVA::MsgLogger Log;

   bool pyIsInitialized = PyIsInitialized();
   if (!pyIsInitialized) {
      Py_Initialize();
   }

    PyGILRAII thePyGILRAII;

   if (!pyIsInitialized) {
      _import_array();
   }

   fMain = PyImport_AddModule("__main__");
   if (!fMain) {
      Log << kFATAL << "Can't import __main__" << Endl;
      Log << Endl;
   }

   fGlobalNS = PyModule_GetDict(fMain);
   if (!fGlobalNS) {
      Log << kFATAL << "Can't init global namespace" << Endl;
      Log << Endl;
   }

   #if PY_MAJOR_VERSION < 3
   //preparing objects for eval
   PyObject *bName =  PyUnicode_FromString("__builtin__");
   // Import the file as a Python module.
   fModuleBuiltin = PyImport_Import(bName);
   if (!fModuleBuiltin) {
      Log << kFATAL << "Can't import __builtin__" << Endl;
      Log << Endl;
   }
   #else
   //preparing objects for eval
   PyObject *bName =  PyUnicode_FromString("builtins");
   // Import the file as a Python module.
   fModuleBuiltin = PyImport_Import(bName);
   if (!fModuleBuiltin) {
      Log << kFATAL << "Can't import builtins" << Endl;
      Log << Endl;
   }
   #endif

   PyObject *mDict = PyModule_GetDict(fModuleBuiltin);
   fEval = PyDict_GetItemString(mDict, "eval");
   fOpen = PyDict_GetItemString(mDict, "open");

   Py_DECREF(bName);
   Py_DECREF(mDict);
   //preparing objects for pickle
   PyObject *pName = PyUnicode_FromString("pickle");
   // Import the file as a Python module.
   fModulePickle = PyImport_Import(pName);
   if (!fModulePickle) {
      Log << kFATAL << "Can't import pickle" << Endl;
      Log << Endl;
   }
   PyObject *pDict = PyModule_GetDict(fModulePickle);
   fPickleDumps = PyDict_GetItemString(pDict, "dump");
   fPickleLoads = PyDict_GetItemString(pDict, "load");

   Py_DECREF(pName);
   Py_DECREF(pDict);
}

//_______________________________________________________________________
void PyMethodBase::PyFinalize()
{
   Py_Finalize();
   if (fEval) Py_DECREF(fEval);
   if (fModuleBuiltin) Py_DECREF(fModuleBuiltin);
   if (fPickleDumps) Py_DECREF(fPickleDumps);
   if (fPickleLoads) Py_DECREF(fPickleLoads);
   if(fMain) Py_DECREF(fMain);//objects fGlobalNS and fLocalNS will be free here
}
void PyMethodBase::PySetProgramName(TString name)
{
   #if PY_MAJOR_VERSION < 3
   Py_SetProgramName(const_cast<char*>(name.Data()));
   #else
   Py_SetProgramName((wchar_t *)name.Data());
   #endif
}

size_t mystrlen(const char* s) { return strlen(s); }
size_t mystrlen(const wchar_t* s) { return wcslen(s); }

//_______________________________________________________________________
TString PyMethodBase::Py_GetProgramName()
{
auto progName = ::Py_GetProgramName();
return std::string(progName, progName + mystrlen(progName));
}
//_______________________________________________________________________
int  PyMethodBase::PyIsInitialized()
{
   if (!Py_IsInitialized()) return kFALSE;
   if (!fEval) return kFALSE;
   if (!fModuleBuiltin) return kFALSE;
   if (!fPickleDumps) return kFALSE;
   if (!fPickleLoads) return kFALSE;
   return kTRUE;
}

void PyMethodBase::Serialize(TString path,PyObject *obj)
{
 if(!PyIsInitialized()) PyInitialize();
 PyObject *file_arg = Py_BuildValue("(ss)", path.Data(),"wb");
 PyObject *file = PyObject_CallObject(fOpen,file_arg);
 PyObject *model_arg = Py_BuildValue("(OO)", obj,file);
 PyObject *model_data = PyObject_CallObject(fPickleDumps , model_arg);

 Py_DECREF(file_arg);
 Py_DECREF(file);
 Py_DECREF(model_arg);
 Py_DECREF(model_data);
}

void PyMethodBase::UnSerialize(TString path,PyObject **obj)
{
 PyObject *file_arg = Py_BuildValue("(ss)", path.Data(),"rb");
 PyObject *file = PyObject_CallObject(fOpen,file_arg);

 PyObject *model_arg = Py_BuildValue("(O)", file);
 *obj = PyObject_CallObject(fPickleLoads , model_arg);

 Py_DECREF(file_arg);
 Py_DECREF(file);
 Py_DECREF(model_arg);
}


////////////////////////////////////////////////////////////////////////////////
/// get all the MVA values for the events of the current Data type
std::vector<Double_t> PyMethodBase::GetMvaValues(Long64_t firstEvt, Long64_t lastEvt, Bool_t logProgress)
{

   if (!fClassifier) ReadModelFromFile();

   Long64_t nEvents = Data()->GetNEvents();
   if (firstEvt > lastEvt || lastEvt > nEvents) lastEvt = nEvents;
   if (firstEvt < 0) firstEvt = 0;
   std::vector<Double_t> values(lastEvt-firstEvt);

   nEvents = values.size();

   UInt_t nvars = Data()->GetNVariables();

   int dims[2];
   dims[0] = nEvents;
   dims[1] = nvars;
   PyArrayObject *pEvent= (PyArrayObject *)PyArray_FromDims(2, dims, NPY_FLOAT);
   float *pValue = (float *)(PyArray_DATA(pEvent));

//    int dims2[2];
//    dims2[0] = 1;
//    dims2[1] = nvars;

   // use timer
   Timer timer( nEvents, GetName(), kTRUE );
   if (logProgress)
      Log() << kINFO<<Form("Dataset[%s] : ",DataInfo().GetName())<< "Evaluation of " << GetMethodName() << " on "
            << (Data()->GetCurrentType()==Types::kTraining?"training":"testing") << " sample (" << nEvents << " events)" << Endl;


   // fill numpy array with events data
   for (Int_t ievt=0; ievt<nEvents; ievt++) {
     Data()->SetCurrentEvent(ievt);
      const TMVA::Event *e = Data()->GetEvent();
      assert(nvars == e->GetNVariables());
      for (UInt_t i = 0; i < nvars; i++) {
         pValue[ievt * nvars + i] = e->GetValue(i);
      }
      // if (ievt%100 == 0)
      //    std::cout << "Event " << ievt << "  type" << DataInfo().IsSignal(e) << " : " << pValue[ievt*nvars] << "  " << pValue[ievt*nvars+1] << "  " << pValue[ievt*nvars+2] << std::endl;
   }

   // pass all the events to Scikit and evaluate the probabilities
   PyArrayObject *result = (PyArrayObject *)PyObject_CallMethod(fClassifier, const_cast<char *>("predict_proba"), const_cast<char *>("(O)"), pEvent);
   double *proba = (double *)(PyArray_DATA(result));

   // the return probabilities is a vector of pairs of (p_sig,p_backg)
   // we ar einterested only in the signal probability
   std::vector<double> mvaValues(nEvents);
   for (int i = 0; i < nEvents; ++i)
      mvaValues[i] = proba[2*i];

   if (logProgress) {
      Log() << kINFO <<Form("Dataset[%s] : ",DataInfo().GetName())<< "Elapsed time for evaluation of " << nEvents <<  " events: "
            << timer.GetElapsedTime() << "       " << Endl;
   }

   Py_DECREF(result);
   Py_DECREF(pEvent);

   return mvaValues;
}

//_______________________________________________________________________
// Helper function to run python code from string in local namespace with
// error handling
// `start` defines the start symbol defined in PyRun_String (Py_eval_input,
// Py_single_input, Py_file_input)
void PyMethodBase::PyRunString(TString code, TString errorMessage, int start) {
   fPyReturn = PyRun_String(code, start, fGlobalNS, fLocalNS);
   if (!fPyReturn) {
      Log() << kWARNING << "Failed to run python code: " << code << Endl;
      Log() << kWARNING << "Python error message:" << Endl;
      PyErr_Print();
      Log() << kFATAL << errorMessage << Endl;
   }
}
