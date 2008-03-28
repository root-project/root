// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_TChi2ExtendedFCN_H_
#define ROOT_TChi2ExtendedFCN_H_

#include "Minuit2/FCNBase.h"

// temporary - should use interface
class TF1;
class TVirtualFitter;

class TChi2ExtendedFitData;

/** 

   Extended Chi2 Fit method. 
   Use errors in X as well, if asymmetric make them symmetric taking the average
*/

class TChi2ExtendedFCN : public ROOT::Minuit2::FCNBase {

public: 

//   // use a param function instead of TF1
  typedef TF1 ModelFunction;



  /**
     construct passing fitter which has ROOT data object and  model function. 
   */
  TChi2ExtendedFCN( const TVirtualFitter & fitter);  

  //Chi2FCN( const DataObject & data, const ModelFunction & func);  


  /**
     construct objective function passing input data and model function.  
   */

  //Chi2FCN( const DataObject & data, const ModelFunction & func);  

  /**
     this class manages the fit data class. Delete it at the end
   */
  ~TChi2ExtendedFCN();

 
  /**
     evaluate objective function 
  */
  double operator()(const std::vector<double>&) const; 


  /**
     return error definition for chi2 should be  1
  */
  double Up() const { return fUp; }

  void SetErrorDef( double up) { fUp = up; }


private: 

  double fUp;
  ModelFunction * fFunc; 
  TChi2ExtendedFitData * fData; 

};


#endif
