// @(#)root/minuit2:$Name:  $:$Id: TBinLikelihoodFCN.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_TBinLikelihoodFCN_H_
#define ROOT_TBinLikelihoodFCN_H_

#include "Minuit/FCNBase.h"


class TF1;
class TVirtualFitter;
class TChi2FitData;

  /**
     Class implementing the standard Chi2 objective function 
   */ 

class TBinLikelihoodFCN : public FCNBase {



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
  double up() const { return fUp; }

  void SetErrorDef( double up) { fUp = up; }

private: 

  double fUp; 
  bool fOwner;
  BinLikelihoodFitData * fData; 
  ModelFunction * fFunc; 

};


#endif
