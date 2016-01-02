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

#include<TMVA/PyMethodBase.h>
#include<TApplication.h>

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <fstream>

using namespace TMVA;

ClassImp(PyMethodBase)

PyObject *PyMethodBase::fModuleBuiltin = NULL;
PyObject *PyMethodBase::fEval = NULL;
PyObject *PyMethodBase::fModulePickle = NULL;
PyObject *PyMethodBase::fPickleDumps = NULL;
PyObject *PyMethodBase::fPickleLoads = NULL;

PyObject *PyMethodBase::fMain = NULL;
PyObject *PyMethodBase::fGlobalNS = NULL;
PyObject *PyMethodBase::fLocalNS = NULL;



//_______________________________________________________________________
PyMethodBase::PyMethodBase(const TString &jobName,
                           Types::EMVA methodType,
                           const TString &methodTitle,
                           DataSetInfo &dsi,
                           const TString &theOption ,
                           TDirectory *theBaseDir): MethodBase(jobName, methodType, methodTitle, dsi, theOption, theBaseDir),
   fClassifier(NULL)
{
   if (!PyIsInitialized()) {
      PyInitialize();
   }
}

//_______________________________________________________________________
PyMethodBase::PyMethodBase(Types::EMVA methodType,
                           DataSetInfo &dsi,
                           const TString &weightFile,
                           TDirectory *theBaseDir): MethodBase(methodType, dsi, weightFile, theBaseDir),
   fClassifier(NULL)
{
   if (!PyIsInitialized()) {
      PyInitialize();
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
void PyMethodBase::PyInitialize()
{
   TMVA::MsgLogger Log;
   if (!PyIsInitialized()) {
      Py_Initialize();
      _import_array();
      import_array();
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
   
   fLocalNS = PyDict_New();
   if (!fMain) {
      Log << kFATAL << "Can't init local namespace" << Endl;
      Log << Endl;
   }
   
   //preparing objects for eval
   PyObject *bName =  PyUnicode_FromString("__builtin__");
   // Import the file as a Python module.
   fModuleBuiltin = PyImport_Import(bName);
   if (!fModuleBuiltin) {
      Log << kFATAL << "Can't import __builtin__" << Endl;
      Log << Endl;
   }
   PyObject *mDict = PyModule_GetDict(fModuleBuiltin);
   fEval = PyDict_GetItemString(mDict, "eval");

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
   fPickleDumps = PyDict_GetItemString(pDict, "dumps");
   fPickleLoads = PyDict_GetItemString(pDict, "loads");

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
   if(fMain) Py_DECREF(fMain);
   if(fGlobalNS) Py_DECREF(fGlobalNS);
   if(fLocalNS) Py_DECREF(fLocalNS);
}
void PyMethodBase::PySetProgramName(TString name)
{
   Py_SetProgramName(const_cast<char *>(name.Data()));
}
//_______________________________________________________________________
TString PyMethodBase::Py_GetProgramName()
{
   return Py_GetProgramName();
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
 
 PyObject *model_arg = Py_BuildValue("(O)", obj);
 PyObject *model_data = PyObject_CallObject(fPickleDumps , model_arg);
 std::ofstream PyData;
//  PyData.open(path.Data(),std::ios_base::binary|std::ios_base::trunc);
 PyData.open(path.Data(),std::ios_base::binary|std::ios_base::trunc);
 PyData << PyBytes_AsString(model_data);
 PyData.close();
 Py_DECREF(model_arg);
 Py_DECREF(model_data);
}

void PyMethodBase::UnSerialize(TString path,PyObject **obj)
{
   std::ifstream PyData;
   std::stringstream PyDataStream;
   std::string PyDataString;

   PyData.open(path.Data(),std::ios_base::binary);
   PyDataStream << PyData.rdbuf();
   PyDataString = PyDataStream.str();
   PyData.close();

   PyObject *model_arg = Py_BuildValue("(s)", PyDataString.c_str());
   *obj = PyObject_CallObject(fPickleLoads , model_arg);
   Py_DECREF(model_arg);
}

      