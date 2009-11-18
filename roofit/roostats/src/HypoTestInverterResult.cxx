// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/**
   HypoTestInverterResult class

   New contributions to this class have been written by Matthias Wolf (error estimation)
**/


// include header file of this class 
#include "RooStats/HypoTestInverterResult.h"

#include "RooStats/HypoTestInverterPlot.h"
#include "RooStats/HybridResult.h"

#include "TF1.h"
#include "TGraphErrors.h"

ClassImp(RooStats::HypoTestInverterResult)

using namespace RooStats;


HypoTestInverterResult::HypoTestInverterResult(const char * name ) :
   SimpleInterval(name),
   fUseCLs(false),
   fUpperLimitError(0)
{
  // default constructor
}


HypoTestInverterResult::HypoTestInverterResult( const char* name,
						const RooRealVar& scannedVariable,
						double cl ) :
   SimpleInterval(name,scannedVariable,-999,999,cl), 
   fUseCLs(false)
{
  // constructor 
   fYObjects.SetOwner();
}


HypoTestInverterResult::~HypoTestInverterResult()
{
   // destructor
   // no need to delete explictly the objects in the TList since the TList owns the objects
}

 
double HypoTestInverterResult::GetXValue( int index ) const
{
  if ( index >= Size() || index<0 ) {
    std::cout << "Problem: You are asking for an impossible array index value\n";
    return -999;
  }

  return fXValues[index];
}

double HypoTestInverterResult::GetYValue( int index ) const
{
  if ( index >= Size() || index<0 ) {
    std::cout << "Problem: You are asking for an impossible array index value\n";
    return -999;
  }

  if (fUseCLs) 
    return ((HybridResult*)fYObjects.At(index))->CLs();
  else 
    return ((HybridResult*)fYObjects.At(index))->AlternatePValue();  // CLs+b
}

double HypoTestInverterResult::GetYError( int index ) const
{
  if ( index >= Size() || index<0 ) {
    std::cout << "Problem: You are asking for an impossible array index value\n";
    return -999;
  }

  if (fUseCLs) 
    return ((HybridResult*)fYObjects.At(index))->CLsError();
  else 
    return ((HybridResult*)fYObjects.At(index))->CLsplusbError();
}

void HypoTestInverterResult::CalculateLimits()
{ 
  // find the 2 objects the closer to the limit and make a linear extrapolation to the target

  double cl = 1-ConfidenceLevel();

  if (Size()<2) {
    std::cout << "not enough points to get the inverted interval\n";
  }

  double v1 = fabs(GetYValue(0)-cl);
  int i1 = 0;
  double v2 = fabs(GetYValue(1)-cl);
  int i2 = 1;

  if (Size()>2)
    for (int i=2; i<Size(); i++) {
      double vt = fabs(GetYValue(i)-cl);
      if ( vt<v1 || vt<v2 ) {
	if ( v1<v2 ) {
	  v2 = vt;
	  i2 = i;
	} else {
	  v1 = vt;
	  i1 = i;
	}
      }
    }

  fLowerLimit = ((RooRealVar*)fParameters.first())->getMin();
  fUpperLimit = GetXValue(i1)+(cl-GetYValue(i1))*(GetXValue(i2)-GetXValue(i1))/(GetYValue(i2)-GetYValue(i1)); // MAYBE TOO MANY GETYVALUE CALLS!

  return;
}


Double_t HypoTestInverterResult::UpperLimitEstimatedError()
{
  // Return an error estimate on the upper limit.  This is the error on
  // either CLs or CLsplusb divided by an estimate of the slope at this
  // point.

  if (Size()<2) {
    std::cout << "not enough points to get the inverted interval\n";
  }

  // Don't calculate an error if we already did so, or if we interpolated
  // the upper limit.
  if (fUpperLimitError > 0) return fUpperLimitError;
 
  // The graph contains the points sorted by their x-value
  HypoTestInverterPlot plot("plot", "", this);
  TGraphErrors* graph = plot.MakePlot();
  double* xs = graph->GetX();
  const double minX = xs[0];
  const double maxX = xs[Size()-1];

  TF1* fct = new TF1("fct", "exp([0] * x + [1] * x**2)", minX, maxX);
  graph->Fit(fct);

  delete fct;
  delete graph;

  // find the object the closest to the limit
  double cl = 1-ConfidenceLevel();
  double v1 = fabs(GetYValue(0)-cl);
  int i1 = 0;
  if (Size()>2)
    for (int i=2; i<Size(); i++) {
      double vt = fabs(GetYValue(i)-cl);
      if ( vt<v1 ) {
	v1 = vt;
	i1 = i;
      }
    }

  double m = fct->Derivative( GetXValue(i1) );

  fUpperLimitError = fabs( GetYError(i1) / m);
  
  return fUpperLimitError;
}
