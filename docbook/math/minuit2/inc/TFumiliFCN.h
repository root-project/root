// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_TFumiliChi2FCN_H_
#define ROOT_TFumiliChi2FCN_H_

#include "Minuit2/FumiliFCNBase.h"

// temporary - should use interface
class TF1;
class TVirtualFitter;

class TChi2FitData;

/** 
    Base Class for implementing Fumili interface
*/


class TFumiliFCN : public ROOT::Minuit2::FumiliFCNBase {

public: 

//   // use a param function instead of TF1
  typedef TF1 ModelFunction;
  typedef TChi2FitData FumiliFitData; 
  


  /**
     construct passing fitter which has ROOT data object and  model function. 
     use a strategy for calculating derivatives 
     strategy = 1   default 2 point formula . Fast but not very precise
     strategy = 2   5 point formula
   */
  TFumiliFCN( const TVirtualFitter & fitter, double up = 1., int strategy = 1, bool sipEmptyBins = true);  


  /**
     this class manages the fit data class. Delete it at the end
   */
  virtual ~TFumiliFCN();

 
  /**
     evaluate objective function 
  */
  virtual double operator()(const std::vector<double>&) const = 0; 


  /**
     evaluate gradient and function elements needed by fumili 
   */
  void EvaluateAll( const std::vector<double> & );  

  /**
     return error definition for chi2 = 1
  */
  double Up() const { return fUp; }

  void SetErrorDef (double up) { fUp = up; }


  /**
     initialize method to set right number of parameters. 
     It is known only when starting  the fit
   */

  void Initialize(unsigned int npar);


protected: 
  
  
  void Calculate_gradient_and_hessian(const std::vector<double> & p);

  void Calculate_numerical_gradient( const std::vector<double> & x, double f0);

 void Calculate_numerical_gradient_of_integral( const std::vector<double> & x1,  const std::vector<double> & x2, double f0);

  // calculate i-th element contribution to objective function
  // abstract - need to be re-implemented by the derived classes§
  virtual void Calculate_element(int i, const FumiliFitData & points, double fval, double & chi2, std::vector<double> & grad,   std::vector<double> & hess ) = 0;

protected: 

  double fUp; 
  FumiliFitData * fData; 
  ModelFunction * fFunc; 
  
//   std::vector<double> fGradient;
//   std::vector<double> fHessian;

  //prameter cache
  std::vector<double>  fParamCache; 
  std::vector<double>  fFunctionGradient; 

  int fStrategy;
};


class TFumiliChi2FCN : public TFumiliFCN {

  public: 
  /**
     construct passing fitter which has ROOT data object and  model function. 
     use a strategy for calculating derivatives 
     strategy = 1   default 2 point formula . Fast but not very precise
     strategy = 2   5 point formula
   */
  TFumiliChi2FCN( const TVirtualFitter & fitter, int strategy = 1) : 
    TFumiliFCN(fitter, 1.0, strategy, true) {}

  virtual ~TFumiliChi2FCN() {}


  /**
     evaluate objective function 
  */
  double operator()(const std::vector<double>&) const; 

protected: 

  virtual void Calculate_element(int i, const TChi2FitData & points, double fval, double & chi2, std::vector<double> & grad, std::vector<double> & hess );
  
};


/** 
    Fumili interface for binned (Poisson) likelihood functions
*/ 

class TFumiliBinLikelihoodFCN : public TFumiliFCN {

  public: 
  /**
     construct passing fitter which has ROOT data object and  model function. 
     use a strategy for calculating derivatives 
     strategy = 1   default 2 point formula . Fast but not very precise
     strategy = 2   5 point formula
   */
  TFumiliBinLikelihoodFCN( const TVirtualFitter & fitter, int strategy = 1)  : 
    TFumiliFCN(fitter, 1.0, strategy, false) {}


  virtual ~TFumiliBinLikelihoodFCN() {}

  /**
     evaluate objective function 
  */
  double operator()(const std::vector<double>&) const; 

  /**
     evaluate chi2 equivalent on the data set
  */
  double Chi2 ( const std::vector<double>&) const; 

protected: 

  virtual void Calculate_element(int i, const TChi2FitData & points, double fval, double & chi2, std::vector<double> & grad,   std::vector<double> & hess );

};


/** 
    Fumili interface for Unbinned  likelihood functions
*/ 
class TFumiliUnbinLikelihoodFCN : public TFumiliFCN {

  public: 
  /**
     construct passing fitter which has ROOT data object and  model function. 
     use a strategy for calculating derivatives 
     strategy = 1   default 2 point formula . Fast but not very precise
     strategy = 2   5 point formula
   */
  TFumiliUnbinLikelihoodFCN( const TVirtualFitter & fitter, int strategy = 1) : 
    TFumiliFCN(fitter, 0.5, strategy, false) {}

  virtual ~TFumiliUnbinLikelihoodFCN() {}

  /**
     evaluate objective function 
  */
  double operator()(const std::vector<double>&) const; 

protected: 

  virtual void Calculate_element(int i, const TChi2FitData & points, double fval, double & chi2, std::vector<double> & grad,   std::vector<double> & hess );

};

#endif
