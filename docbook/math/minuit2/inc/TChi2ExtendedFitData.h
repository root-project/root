// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_TChi2ExtendedFitData_H_
#define ROOT_TChi2ExtendedFitData_H_

#include "RConfig.h"
#include <vector>
#include "TChi2FitData.h"

// class TObject; 
class TVirtualFitter;
class TGraph;

/**
   class holding the data of the fit . 
  For extended chi2 fits which contains also error in x (asymmetrics) 
  the data are: 

  coords() , value , sigma Y, sigma X_low sigma X_up
*/


class TChi2ExtendedFitData : public TChi2FitData { 


public: 

  typedef  std::vector<double> CoordData; 


  /** 
      construct the Fit data object
  */
  TChi2ExtendedFitData() {}

  TChi2ExtendedFitData(const TVirtualFitter & fitter); 

  virtual ~TChi2ExtendedFitData() {}

  double ErrorY(unsigned int i) const { return fErrorsY[i]; }

  double ErrorXLow(unsigned int i) const { return fErrorsXLow[i]; }

  double ErrorXUp(unsigned int i) const { return fErrorsXUp[i]; }


protected:


  void GetExtendedFitData(const TGraph * graph, const TF1 * func, const  TVirtualFitter * fitter); 

  void SetDataPoint(  const CoordData & x, double y, double errorY, double errorXlow, double errorXup );
  

private: 

  std::vector<double> fErrorsY;
  std::vector<double> fErrorsXLow;
  std::vector<double> fErrorsXUp;
  // to do add asymmetric error in y 
  
};

#endif
