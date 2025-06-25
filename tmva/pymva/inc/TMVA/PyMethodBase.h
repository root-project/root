// @(#)root/tmva/pymva $Id$
// Authors: Omar Zapata, Lorenzo Moneta, Sergei Gleyzer 2015, Stefan Wunsch 2017

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


#include "TMVA/MethodBase.h"
#include "TMVA/Types.h"

#include "Rtypes.h"
#include "TString.h"
#include <vector>

class TFile;
class TGraph;
class TTree;
class TDirectory;
class TSpline;
class TH1F;
class TH1D;

#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#define Py_single_input 256
#endif

namespace TMVA {

   class Ranking;
   class PDF;
   class TSpline1;
   class MethodCuts;
   class MethodBoost;
   class DataSetInfo;

   /// Function to find current Python executable
   /// used by ROOT
   /// If "Python3" is installed, return "python3"
   TString Python_Executable();

   /// Virtual base class for all TMVA method based on Python
   class PyMethodBase : public MethodBase {

      friend class Factory;
   public:

      // default constructur
      PyMethodBase(const TString &jobName,
                   Types::EMVA methodType,
                   const TString &methodTitle,
                   DataSetInfo &dsi,
                   const TString &theOption = "");

      // constructor used for Testing + Application of the MVA, only (no training),
      // using given weight file
      PyMethodBase(Types::EMVA methodType,
                   DataSetInfo &dsi,
                   const TString &weightFile);

      // default destructur
      virtual ~PyMethodBase();
      //basic python related function
      static void PyInitialize();
      static int  PyIsInitialized();
      static void PyFinalize();

      PyObject *Eval(TString code); // required to parse booking options from string to pyobjects
      static void Serialize(TString file,PyObject *classifier);
      static Int_t UnSerialize(TString file,PyObject** obj);

      void     Train() override = 0;
      // options treatment
      void     Init() override           = 0;
      void     DeclareOptions() override = 0;
      void     ProcessOptions() override = 0;
      // create ranking
      const Ranking *CreateRanking() override = 0;

      Double_t GetMvaValue(Double_t *errLower = nullptr, Double_t *errUpper = nullptr) override = 0;

      Bool_t HasAnalysisType(Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets) override = 0;
   protected:
      // the actual "weights"
      void AddWeightsXMLTo(void *parent) const override = 0;
      void ReadWeightsFromXML(void *wghtnode) override = 0;
      void ReadWeightsFromStream(std::istream &) override = 0; // backward compatibility
      void ReadWeightsFromStream(TFile &) override {} // backward compatibility

      virtual void ReadModelFromFile() = 0;

      // signal/background classification response for all current set of data
      std::vector<Double_t> GetMvaValues(Long64_t firstEvt = 0, Long64_t lastEvt = -1, Bool_t logProgress = false) override = 0;

   protected:
      PyObject *fModule; // Module to load
      PyObject *fClassifier; // Classifier object

      PyObject *fPyReturn; // python return data

   protected:
      void PyRunString(TString code, TString errorMessage="Failed to run python code", int start=Py_single_input); // runs python code from string in local namespace with error handling

   private:
      static PyObject *fModuleBuiltin;
      static PyObject *fEval; // eval funtion from python
      static PyObject *fOpen; // open function for files

   protected:
      static PyObject *fModulePickle; // Module for model persistence
      static PyObject *fPickleDumps; // Function to dumps PyObject information into string
      static PyObject *fPickleLoads; // Function to load PyObject information from string

      static PyObject *fMain; // module __main__ to get namespace local and global
      static PyObject *fGlobalNS; // global namesapace
      PyObject *fLocalNS; // local namesapace

   public:
      static void PyRunString(TString code, PyObject *globalNS, PyObject* localNS); // Overloaded static Python utlity function for running Python code
      static const char* PyStringAsString(PyObject *string); // Python Utility function for converting a Python String object to const char*
      static std::vector<size_t> GetDataFromTuple(PyObject *tupleObject);  // Function casts Python Tuple object into vector of size_t
      static std::vector<size_t> GetDataFromList(PyObject *listObject);    // Function casts Python List object into vector of size_t
      static PyObject* GetValueFromDict(PyObject* dict, const char* key);  // Function to check for a key in dict and return the associated value if present
      ClassDefOverride(PyMethodBase, 0) // Virtual base class for all TMVA method

   };

} // namespace TMVA

#endif
