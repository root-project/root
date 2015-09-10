// @(#)root/tmva/pymva $Id$
// Authors: Omar Zapata, Lorenzo Moneta, Sergei Gleyzer 2015

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : PyMethodBase                                                          *
 * Web    : http://oproject.org                                                   *
 *                                                                                *
 * Description:                                                                   *
 *      Virtual base class for all MVA method based on Python                     *
 *                                                                                *
 **********************************************************************************/

#ifndef ROOT_TMVA_PyMethodBase
#define ROOT_TMVA_PyMethodBase

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// PyMethodBase                                                               //
//                                                                            //
// Virtual base class for all TMVA method based on Python/scikit-learn        //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

class TGraph;
class TTree;
class TDirectory;
class TSpline;
class TH1F;
class TH1D;

namespace TMVA {

   class Ranking;
   class PDF;
   class TSpline1;
   class MethodCuts;
   class MethodBoost;
   class DataSetInfo;

   class PyMethodBase : public MethodBase {

      friend class Factory;
   public:

      // default constructur
      PyMethodBase(const TString &jobName,
                   Types::EMVA methodType,
                   const TString &methodTitle,
                   DataSetInfo &dsi,
                   const TString &theOption = "",
                   TDirectory *theBaseDir = 0);

      // constructor used for Testing + Application of the MVA, only (no training),
      // using given weight file
      PyMethodBase(Types::EMVA methodType,
                   DataSetInfo &dsi,
                   const TString &weightFile,
                   TDirectory *theBaseDir = 0);

      // default destructur
      virtual ~PyMethodBase();
      //basic python related function
      static void PyInitialize();
      static int  PyIsInitialized();
      static void PyFinalize();
      static void PySetProgramName(TString name)
      {
         Py_SetProgramName(const_cast<char *>(name.Data()));
      }
      static TString Py_GetProgramName()
      {
         return Py_GetProgramName();
      }
      static PyObject *Eval(TString code);//required to parse booking options from string to pyobjects


      virtual void     Train() = 0;
      // options treatment
      virtual void     Init()           = 0;
      virtual void     DeclareOptions() = 0;
      virtual void     ProcessOptions() = 0;
      // create ranking
      virtual const Ranking *CreateRanking() = 0;

      virtual Double_t GetMvaValue(Double_t *errLower = 0, Double_t *errUpper = 0) = 0;

      Bool_t HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets) = 0;
   protected:
      // the actual "weights"
      virtual void AddWeightsXMLTo(void *parent) const = 0;
      virtual void ReadWeightsFromXML(void *wghtnode) = 0;
      virtual void ReadWeightsFromStream(std::istream &) = 0;        // backward compatibility
      virtual void ReadWeightsFromStream(TFile &) {}                 // backward compatibility

   protected:
      PyObject *fModule;//Module to load
      PyObject *fClassifier;//Classifier object

      PyArrayObject *fTrainData;
      PyArrayObject *fTrainDataWeights;//array of weights
      PyArrayObject *fTrainDataClasses;//array with sig/bgk class
   private:

      static PyObject *fModuleBuiltin;
      static PyObject *fEval;
   protected:
      static PyObject *fModulePickle;//Module for model persistence
      static PyObject *fPickleDumps;//Function to dumps PyObject information into string
      static PyObject *fPickleLoads;//Function to load PyObject information from string

      ClassDef(PyMethodBase, 0) // Virtual base class for all TMVA method

   };
} // namespace TMVA

#endif


