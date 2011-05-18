// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_TBinLikelihoodFCN_H_
#define ROOT_TBinLikelihoodFCN_H_

#include "Minuit2/FCNBase.h"


class TF1;
class TVirtualFitter;
class TChi2FitData;

  /**
     Class implementing the standard Chi2 objective function 
   */ 

class TBinLikelihoodFCN : public ROOT::Minuit2::FCNBase {



public: 

//   // use a param function instead of TF1
  typedef TF1 ModelFunction;
  // use now same data as chi2 fit data
  typedef TChi2FitData BinLikelihoodFitData;



  /**
     construct passing fitter which has ROOT data object and  model function. 
   */
  TBinLikelihoodFCN( const TVirtualFitter & fitter);  

  /**
     constructor passing data and function 
     In this case does not own the data
   */
  TBinLikelihoodFCN( TChi2FitData * data, ModelFunction * func) : 
    fUp(0.5), fOwner(false), fData(data), fFunc(func) {}


  /**
     this class manages the fit data class. Delete it at the end
   */
  ~TBinLikelihoodFCN();

 
  /**
     evaluate objective function 
  */
  double operator()(const std::vector<double>&) const; 


  /**
     return error definition for likelihood = 0.5
  */
  double Up() const { return fUp; }

  void SetErrorDef( double up) { fUp = up; }

  /**
     evaluate chi2 equivalent on the data set
  */
  double Chi2 ( const std::vector<double>&) const; 


private: 

  double fUp; 
  bool fOwner;
  BinLikelihoodFitData * fData; 
  ModelFunction * fFunc; 

};


#endif
