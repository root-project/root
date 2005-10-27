// @(#)root/minuit2:$Name:  $:$Id: TFitterMinuit.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_TFitterMinuit_H_
#define ROOT_TFitterMinuit_H_

#ifndef ROOT_TVirtualFitter
#include "TVirtualFitter.h"
#endif

#include "Minuit/MnUserParameterState.h"
#include "Minuit/MinosError.h"
#include "Minuit/ModularFunctionMinimizer.h"
#include "Minuit/FumiliMinimizer.h"
#include "TFcnAdapter.h"

/**
    TVirtualFitter implementation for new C++ Minuit
*/

class FunctionMinimum;

class TFitterMinuit : public TVirtualFitter {

public:

  // enumeration specifying the minimizers
  enum EMinimizerType { 
    kMigrad, 
    kSimplex, 
    kCombined, 
    kScan,
    kFumili
  };
   

  TFitterMinuit();

  TFitterMinuit(Int_t maxpar);

  virtual ~TFitterMinuit();

public:

  // inherited interface
   virtual Double_t  Chisquare(Int_t npar, Double_t *params) const;
   virtual void      Clear(Option_t *option="");
   virtual Int_t     ExecuteCommand(const char *command, Double_t *args, Int_t nargs);
   virtual void      FixParameter(Int_t ipar);
   virtual Double_t *GetCovarianceMatrix() const;
   virtual Double_t  GetCovarianceMatrixElement(Int_t i, Int_t j) const;
   virtual Int_t     GetErrors(Int_t ipar,Double_t &eplus, Double_t &eminus, Double_t &eparab, Double_t &globcc) const;
   //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
   virtual Int_t     GetNumberTotalParameters() const;
   virtual Int_t     GetNumberFreeParameters() const;

   virtual Double_t  GetParError(Int_t ipar) const;
   virtual Double_t  GetParameter(Int_t ipar) const;
   virtual Int_t     GetParameter(Int_t ipar,char *name,Double_t &value,Double_t &verr,Double_t &vlow, Double_t &vhigh) const;
   virtual Int_t     GetStats(Double_t &amin, Double_t &edm, Double_t &errdef, Int_t &nvpar, Int_t &nparx) const;
   virtual Double_t  GetSumLog(Int_t i);

   //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
   virtual Bool_t    IsFixed(Int_t ipar) const ;

   virtual void      PrintResults(Int_t level, Double_t amin) const;
   virtual void      ReleaseParameter(Int_t ipar);
   virtual void      SetFitMethod(const char *name);
   virtual Int_t     SetParameter(Int_t ipar,const char *parname,Double_t value,Double_t verr,Double_t vlow, Double_t vhigh);

   virtual void      SetFCN(void (*fcn)(Int_t &, Double_t *, Double_t &f, Double_t *, Int_t) );
   // this for CINT (interactive functions)
   virtual void      SetFCN(void * );

  // set FCN using Minuit interface
  // you pass to the class ownership of FCNBase pointer

  virtual void SetMinuitFCN(  FCNBase * f);

  // methods needed by derived classes 
  virtual const MnUserParameterState & State() const { return fState; } 

  virtual const FCNBase * GetMinuitFCN() const { return fMinuitFCN; } 

  virtual const ModularFunctionMinimizer * GetMinimizer() const { return fMinimizer; }


  // additional abstract methods to be implemented by derived classes 
  virtual int Minimize(  int nfcn = 0, double edmval = 0.1);

  int GetStrategy() { return fStrategy; }

  int PrintLevel() { return fDebug; }

  void SetStrategy( int stra) { fStrategy = stra; } 

  void SetPrintLevel(int level ) { fDebug = level; } 

  // set minimum tolerance to avoid having clients (as TGraf::Fit) setting tolerances too small
  void SetMinimumTolerance(double mintol) { fMinTolerance = mintol; }

  double MinimumTolerance() const { return fMinTolerance; }
  

protected: 

  // method to set internal data (no copying involved so - make protected )

  virtual MnUserParameterState & State() { return fState; }

  virtual void SetMinimizer( ModularFunctionMinimizer * m) { fMinimizer = m; }


  // re -implemented in derived classes 

  virtual void CreateMinimizer(EMinimizerType = kMigrad ); 

  // functions to create FCN - re-implemented in derived class (GFumili)

  virtual void CreateChi2FCN(); 

  virtual void CreateChi2ExtendedFCN(); 

  virtual void CreateBinLikelihoodFCN();

  virtual void CreateUnbinLikelihoodFCN() {}

  // internal function to perform the actual minimization (could be implemented by derived classes)
  virtual FunctionMinimum DoMinimization( int nfcn = 0, double edmval = 0.1);

  // internal funcition to study Function minimum results
  // return 0 if function minimum is OK or an error code

  virtual int ExamineMinimum(const FunctionMinimum & );
  
  virtual void Initialize();
   
private:

  double fErrorDef;
  double fEDMVal;
  bool fGradient;

  MnUserParameterState fState;
  std::vector<MinosError> theMinosErrors;
  ModularFunctionMinimizer * fMinimizer;
  FCNBase * fMinuitFCN;
  int fDebug;
  int fStrategy;
  double fMinTolerance;


  ClassDef(TFitterMinuit,1)
};

R__EXTERN TFitterMinuit* gMinuit2;


#endif //ROOT_TFitterMinuit_H_
