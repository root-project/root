/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooMCMC.h,v 1.00 2017/14/11 11:13:42

 * Author:                                                                  *
 *   OD, Oliver Dahme, University of Zurich, o.dahme@cern.ch      *
 *
 *****************************************************************************/
#ifndef ROO_MCMC
#define ROO_MCMC

#include "TObject.h"
#include <vector>
#include "TFile.h"

class RooAbsReal ;
class RooFitResult ;
class RooArgList ;
class RooAbsArg ;
class TVirtualFitter ;
class TH2D ;
class TH1F ;
class RooPlot ;
class TFile;
class TMultiGraph;
class TGraph;

class RooMCMC : public TObject {
public:

  RooMCMC(RooAbsReal& function) ;
  virtual ~RooMCMC() ;
   void setOffsetting(Bool_t flag) ;

  Int_t mcmc(size_t npoints, size_t cutoff, const char* errorstrategy = "gaus"); //minimizes function, cutoff points and calculates errors, "gaus" for symetric ones, and "interval" for asymetric ones.
  TGraph* getProfile(const char* name, Bool_t cutoff = kTRUE);// returns a profile of function for a parameter, including or excluding cutoff-points
  TMultiGraph* getWalkDis(const char* name, Bool_t cutoff = kTRUE);// returns the walk distribution for a parameter, including or excluding cutoff-points
  TH1F*   getWalkDisHis(const char* name,  Int_t nbinsx, Bool_t cutoff = kTRUE); // returns histogram of the walk distribution including or excluding cutoff-points
  TH2D*   getCornerPlot(const char* name1, const char* name2, Int_t nbinsx, Int_t nbinsy, Bool_t cutoff = kTRUE);
  Int_t changeCutoff(Int_t newCutoff); // returns a scatter plot to check for correlations between two parameters
  Int_t printError(const char* name, Double_t conf = 0.682); //prints symetric errors for a parameter at a defined confidence level
  Int_t saveCandidatesAs(const char* name); // saves all points of the function into a file
  Int_t getPercentile(const char* name, Double_t conf = 0.682); // prints asymetric errors at a defined confidence level
  Int_t getGausErrors(); // prints symetric errors for all parameters and the correlation coefficients
  Int_t setPrintLevel(Int_t newLevel) ; // set level of output (not implemented)

  static void cleanup() ; //empty all point lists
  void setSeed(Double_t seed)
  {
    _seed = seed;
  }; //set seed for random generator
  void setAlphaStar(Double_t newAlpha) {
    _alphaStar = newAlpha;
  }; // change forced acceptance rate, not recommended

  inline size_t getNPar() const { return _nPar ; } // returns number of parameters
  std::vector<const char*> getNames(); // returns vector of the parameter names

 protected:
   virtual Bool_t setPdfParamVal(Int_t index, Double_t value, Bool_t verbose=kFALSE) ; // set pdf parameter values
   void setPdfParamErr(Int_t index, Double_t value) ; // set pdf prameter errors
     void updateFloatVec() ; // updates float vector
     void sortPointList(const char* name1) ; // sorts point list according to a parameter
     Int_t getIndex(const char* name) ; // returns internal index of a parameter
     Double_t getMinList(const char* name); // returns minimal value of a parameter
     Double_t getMaxList(const char* name); // returns maximum value of a parameter
     void setFileName(const TString name)
     {
       _fileName = name;
     }; // set name of outputfile

 private:
   Int_t       _printLevel ; //level of printing
   Int_t       _status ; //status level
   size_t       _nPar ; //number of parameters
   Int_t       _cutoff;//number of points before cutoff
   RooArgList* _floatParamList ; //parameter list
   std::vector<RooAbsArg*> _floatParamVec ; //vector of parameter list
   RooArgList* _initFloatParamList ; //initial parameter list
   RooArgList* _constParamList ; //list of constant parameters
   RooArgList* _initConstParamList ; //intial list of constant parameters
   RooArgList*  _bestParamList; //list of parameters with lowest function value
   RooAbsReal* _func ; //function to be minimized
   std::vector<RooArgList*> _pointList; //list of monte carlo markov chain points
   std::vector<RooArgList*> _cutoffList; //list of points after cutoff
   std::vector<RooArgList*> _sortPointList; //sorted list of points
   TString _fileName; //name of output file
   Bool_t      _verbose ; //turn verbosity on or off
   Bool_t     _gaus; //turn gaus errors on or off in mcmc()
   Bool_t     _interval; //turn asymetric errors on or off in mcmc()
   Double_t   _seed = 0; //seed for random numbers generator
   Double_t  _alphaStar = 0.234; //forced acceptance rate
   static TVirtualFitter *_theFitter ; //fitter function
   RooMCMC(const RooMCMC&) ;

   ClassDef(RooMCMC,0) // RooFit minimizer based on Monte Carlo Markov Chain
} ;


#endif
