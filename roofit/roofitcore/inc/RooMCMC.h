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

  Int_t mcmc(size_t npoints, size_t cutoff, const char* errorstrategy = "gaus");
  TGraph* getProfile(const char* name, Bool_t cutoff = kTRUE);
  TMultiGraph* getWalkDis(const char* name, Bool_t cutoff = kTRUE);
  TH1F*   getWalkDisHis(const char* name,  Int_t nbinsx, Bool_t cutoff = kTRUE);
  TH2D*   getCornerPlot(const char* name1, const char* name2, Int_t nbinsx, Int_t nbinsy, Bool_t cutoff = kTRUE);
  Int_t changeCutoff(Int_t newCutoff);
  Int_t printError(const char* name, Double_t conf = 0.682);
  Int_t saveCandidatesAs(const char* name);
  //Int_t saveCornerPlotAs(const char* pngname);
  Int_t getPercentile(const char* name, Double_t conf = 0.682);
  Int_t getGausErrors();
  Int_t setPrintLevel(Int_t newLevel) ;

  static void cleanup() ;
  void setSeed(Double_t seed)
  {
    _seed = seed;
  };
  void setAlphaStar(Double_t newAlpha) {
    _alphaStar = newAlpha;
  };

  inline size_t getNPar() const { return _nPar ; }
  std::vector<const char*> getNames();

 protected:
   virtual Bool_t setPdfParamVal(Int_t index, Double_t value, Bool_t verbose=kFALSE) ;
   void setPdfParamErr(Int_t index, Double_t value) ;
     void updateFloatVec() ;
     void sortPointList(const char* name1) ;
     Int_t getIndex(const char* name) ;
     Double_t getMinList(const char* name);
     Double_t getMaxList(const char* name);
     void setFileName(const TString name)
     {
       _fileName = name;
     };

 private:
   Int_t       _printLevel ;
   Int_t       _status ;
   size_t       _nPar ;
   Int_t       _cutoff;
   RooArgList* _floatParamList ;
   std::vector<RooAbsArg*> _floatParamVec ;
   RooArgList* _initFloatParamList ;
   RooArgList* _constParamList ;
   RooArgList* _initConstParamList ;
   RooArgList*  _bestParamList;
   RooAbsReal* _func ;
   std::vector<RooArgList*> _pointList;
   std::vector<RooArgList*> _cutoffList;
   std::vector<RooArgList*> _sortPointList;
   TString _fileName;
   Bool_t      _verbose ;
   Bool_t     _gaus;
   Bool_t     _interval;
   Double_t   _seed = 0;
   Double_t  _alphaStar = 0.234;
   static TVirtualFitter *_theFitter ;
   RooMCMC(const RooMCMC&) ;

   ClassDef(RooMCMC,0) // RooFit minimizer based on Monte Carlo Markov Chain
} ;


#endif
