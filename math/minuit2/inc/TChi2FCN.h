// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_TChi2FCN_H_
#define ROOT_TChi2FCN_H_

#include "Minuit2/FCNBase.h"


class TF1;
class TVirtualFitter;

class TChi2FitData;

  /**
     Class implementing the standard Chi2 objective function 
   */ 

class TChi2FCN : public ROOT::Minuit2::FCNBase {



public: 

//   // use a param function instead of TF1
  typedef TF1 ModelFunction;



  /**
     construct passing fitter which has ROOT data object and  model function. 
   */
  TChi2FCN( const TVirtualFitter & fitter);  

  /**
     constructor passing data and function 
     In this case does not own the data
   */
  TChi2FCN( TChi2FitData * data, ModelFunction * func) : 
    fUp(1.0), fOwner(false), fData(data), fFunc(func) {}


  /**
     construct objective function passing input data and model function.  
   */

  //TChi2FCN( const DataObject & data, const ModelFunction & func);  

  /**
     this class manages the fit data class. Delete it at the end
   */
  ~TChi2FCN();

 
  /**
     evaluate objective function 
  */
  double operator()(const std::vector<double>&) const; 


  /**
     return error definition for chi2 = 1
  */
  double Up() const { return fUp; }

  void SetErrorDef( double up) { fUp = up; }

protected:


private: 

  double fUp;
  bool fOwner;
  // has to be mutable since I call non const methods (as SetParameters)
  TChi2FitData * fData; 
  mutable ModelFunction * fFunc; 

};


#endif
