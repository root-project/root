// @(#)root/minuit2:$Id$
// Author: L. Moneta    10/2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 ROOT Foundation,  CERN/PH-SFT                   *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_TChi2FitData_H_
#define ROOT_TChi2FitData_H_

#include "RConfig.h"
#include <vector>

// class TObject; 
class TVirtualFitter;
class TH1;
class TF1;
class TGraph;
class TGraph2D;
class TMultiGraph;

/**
   class holding the data of the fit . 
  For chi2 fits the data are: 

  coords() , value, sigmas
*/


class TChi2FitData { 


public: 

  typedef  std::vector<double> CoordData; 


  /** 
      construct the Fit data object
  */
  
   TChi2FitData() : fSize(0), fSkipEmptyBins(false), fIntegral(false) {}

  TChi2FitData(const TVirtualFitter & fitter,  bool skipEmptyBins = true); 

  virtual ~TChi2FitData() {}

  unsigned int Size() const { return fSize; } 

  const CoordData & Coords(unsigned int i) const { return fCoordinates[i]; }

  double Value(unsigned int i) const { return fValues[i]; }

  double InvError(unsigned int i) const { return fInvErrors[i]; }

  bool UseIntegral() const { return fIntegral; }

  bool SkipEmptyBins() const { return fSkipEmptyBins; }


protected:

  virtual void GetFitData(const TH1 * hfit, const TF1 * func, const TVirtualFitter * hfitter); 

  void GetFitData(const TGraph * graph, const TF1 * func, const TVirtualFitter * hfitter); 

  void GetFitData(const TGraph2D * graph, const TF1 * func, const TVirtualFitter * hfitter); 

  void GetFitData(const TMultiGraph * graph, const TF1 * func, const TVirtualFitter * hfitter); 

  void SetDataPoint(  const CoordData & x, double y, double error );
  

protected: 

  unsigned int fSize;
  bool fSkipEmptyBins;
  bool fIntegral;
  std::vector<double> fInvErrors;
  std::vector<double> fValues;
  std::vector<CoordData> fCoordinates;

  
};

#endif
