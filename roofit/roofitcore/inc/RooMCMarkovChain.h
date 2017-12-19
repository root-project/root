/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooMCMarkovChain.h,v 1.00 2017/14/11 11:13:42

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

class RooMCMarkovChain : public TObject {
public:

  RooMCMarkovChain(RooAbsReal& function) ;
  virtual ~RooMCMarkovChain() ;
  /** \class RooMCMarkovChain
    RooMCMarkovChain is used as the RooMinuit class except that it is using a Monte Carlo Markov Chain as a minimizer.A tutorial can be found in the roofit section where a basic comparison with Minuit can be performed.
  */

  ///enable or disable Offesting
  void setOffsetting(Bool_t flag) ;

  ///minimizes function, cutoff points and calculates errors, "gaus" for symetric ones, and "interval" for asymetric ones.
  Int_t mcmc(size_t npoints, size_t cutoff, const char* errorstrategy = "gaus");

  ///returns a profile of function for a parameter, including or excluding cutoff-points
  TGraph* getProfile(const char* name, Bool_t cutoff = kTRUE);

  ///returns the walk distribution for a parameter, including or excluding cutoff-points
  TMultiGraph* getWalkDis(const char* name, Bool_t cutoff = kTRUE);

  ///returns histogram of the walk distribution including or excluding cutoff-points
  TH1F*   getWalkDisHis(const char* name,  Int_t nbinsx, Bool_t cutoff = kTRUE);

  ///returns a scatter plot to check for correlations between two parameters
  TH2D*   getCornerPlot(const char* name1, const char* name2, Int_t nbinsx, Int_t nbinsy, Bool_t cutoff = kTRUE);
  Int_t changeCutoff(Int_t newCutoff);

  ///prints symetric errors for a parameter at a defined confidence level
  Int_t printError(const char* name, Double_t conf = 0.682);

  ///saves all points of the function into a file
  Int_t saveCandidatesAs(const char* name);

  ///prints asymetric errors at a defined confidence level
  Int_t getPercentile(const char* name, Double_t conf = 0.682);

  ///prints symetric errors for all parameters and the correlation coefficients
  Int_t getGausErrors();

  ///set level of output (not implemented)
  Int_t setPrintLevel(Int_t newLevel) ;

  ///empty all point lists
  void cleanup() ;

  ///set seed for random generator
  void setSeed(Double_t seed);

  ///change forced acceptance rate, not recommended
  void setAlphaStar(Double_t newAlpha);

  ///returns number of parameters
  size_t getNPar();

  ///returns vector of the parameter names
  std::vector<const char*> getNames();

 protected:

   ///set pdf parameter values
   virtual Bool_t setPdfParamVal(Int_t index, Double_t value, Bool_t verbose=kFALSE) ;

   ///set pdf prameter errors
   void setPdfParamErr(Int_t index, Double_t value) ;

   ///updates float vector
   void updateFloatVec() ;

   ///sorts point list according to a parameter
   void sortPointList(const char* name1) ;

   ///returns internal index of a parameter
   Int_t getIndex(const char* name) ;

   ///returns minimal value of a parameter
   Double_t getMin(const char* name);

   ///returns maximum value of a parameter
   Double_t getMax(const char* name);

 private:
   Int_t       _printLevel ; ///< level of printing
   Int_t       _status ; ///< status level
   size_t       _nPar ; ///< number of parameters
   Int_t       _cutoff; ///< number of points before cutoff
   RooArgList* _floatParamList ; ///< parameter list
   std::vector<RooAbsArg*> _floatParamVec ; ///< vector of parameter list
   RooArgList* _initFloatParamList ; ///< initial parameter list
   RooArgList* _constParamList ; ///< list of constant parameters
   RooArgList* _initConstParamList ; ///< intial list of constant parameters
   TVectorD*  _bestParamList; ///< list of parameters with lowest function value
   RooAbsReal* _func ; ///< function to be minimized
   std::vector<TVectorD*> _pointList; ///< list of monte carlo markov chain points
   std::vector<const char*> _nameList;
   std::vector<TVectorD*> _cutoffList; ///< list of points after cutoff
   std::vector<TVectorD*> _sortPointList; ///< sorted list of points
   Bool_t      _verbose ; ///< turns verbosity on or off
   Bool_t     _gaus; ///< turns gaus errors on or off in mcmc()
   Bool_t     _interval; ///< turns asymetric errors on or off in mcmc()
   Double_t   _seed = 0; ///< seed for random numbers generator
   Double_t  _alphaStar = 0.234; ///< forced acceptance rate
   static TVirtualFitter *_theFitter ; ///< fitter function

   RooMCMarkovChain(const RooMCMarkovChain&) ;

   ClassDef(RooMCMarkovChain,0) // RooFit minimizer based on Monte Carlo Markov Chain
} ;


#endif
