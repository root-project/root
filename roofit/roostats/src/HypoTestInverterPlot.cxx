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
   HypoTestInverterPlot class
**/

// include other header files
#include "RooStats/HybridResult.h"

// include header file of this class 
#include "RooStats/HypoTestInverterPlot.h"
#include "RooStats/HypoTestInverterResult.h"

#include "TGraphErrors.h"

ClassImp(RooStats::HypoTestInverterPlot)

using namespace RooStats;



HypoTestInverterPlot::HypoTestInverterPlot( const char* name,
					    const char* title,
					    HypoTestInverterResult* results ) :
  TNamed( TString(name), TString(title) ),
  fResults(results)
{
  // constructor
}


TGraphErrors* HypoTestInverterPlot::MakePlot()
{
  const int nEntries = fResults->ArraySize();

  std::vector<Double_t> xArray(nEntries);
  std::vector<Double_t> yArray(nEntries);
  std::vector<Double_t> yErrArray(nEntries);
  for (int i=0; i<nEntries; i++) {
    xArray[i] = fResults->GetXValue(i);
    yArray[i] = fResults->GetYValue(i);
    yErrArray[i] = fResults->GetYError(i);
  }
  
  // sort the arrays based on the x values (using Gnome-sort algorithm)
  if (nEntries>1) {
    int i=1;
    int j=2;
    while ( i<nEntries ) {
      if ( i==0 || xArray[i-1] <= xArray[i] ) {
	i=j;
	j++;
      } else {
	double tmp = xArray[i-1];
	xArray[i-1] = xArray[i];
	xArray[i] = tmp;
	tmp = yArray[i-1];
	yArray[i-1] = yArray[i];
	yArray[i] = tmp;
	tmp = yErrArray[i-1];
	yErrArray[i-1] = yErrArray[i];
	yErrArray[i] = tmp;
	i--;
      }
    }
  }

  TGraphErrors* graph = new TGraphErrors(nEntries,&xArray.front(),&yArray.front(),0,&yErrArray.front());
  graph->SetMarkerStyle(kFullDotMedium);
  return graph;
}

HypoTestInverterPlot::~HypoTestInverterPlot()
{
  // destructor
}
