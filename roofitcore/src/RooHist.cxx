/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHist.cc,v 1.3 2001/04/22 18:15:32 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   30-Nov-2000 DK Created initial version
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// A RooHist is a graphical representation of binned data based on the
// TGraphAsymmErrors class. Error bars are calculated using either Poisson
// or Binomial statistics.

#include "BaBar/BaBar.hh"

#include "RooFitCore/RooHist.hh"
//#include "RooFitTools/RooMath.hh"

#include "TH1.h"
#include <iostream.h>
#include <iomanip.h>

ClassImp(RooHist)

static const char rcsid[] =
"$Id: RooHist.cc,v 1.3 2001/04/22 18:15:32 david Exp $";

RooHist::RooHist(Double_t nSigma) :
  TGraphAsymmErrors(), _nSigma(nSigma)
{
  // Create an empty histogram that can be filled with the addBin()
  // and addAsymmetryBin() methods. Use the optional parameter to
  // specify the confidence level in units of sigma to use for
  // calculating error bars.

  initialize();
}

RooHist::RooHist(const TH1 &data, Double_t nSigma) :
  TGraphAsymmErrors(), _nSigma(nSigma)
{
  // Create a histogram from the contents of the specified TH1 object.
  // Error bars are calculated using Poisson statistics.  Prints a
  // warning and rounds any bins with non-integer contents.  Use the
  // optional parameter to specify the confidence level in units of
  // sigma to use for calculating error bars.

  initialize();
  // copy the input histogram's name and title
  SetName(data.GetName());
  SetTitle(data.GetTitle());
  // TH1::GetYaxis() is not const (why!?)
  setYAxisLabel(const_cast<TH1&>(data).GetYaxis()->GetTitle());
  // initialize our contents from the input histogram's contents
  Int_t nbin= data.GetNbinsX();
  for(Int_t bin= 1; bin <= nbin; bin++) {
    Axis_t x= data.GetBinCenter(bin);
    Stat_t y= data.GetBinContent(bin);
    addBin(x,roundBin(y));
  }
}

void RooHist::initialize() {
  // Perform common initialization for all constructors.

  SetMarkerStyle(8);
}

Int_t RooHist::roundBin(Stat_t y) {
  // Return the nearest positive integer to the input value
  // and print a warning if an adjustment is required.

  if(y < 0) {
    cout << fName << "::roundBin: rounding negative bin contents to zero: " << y << endl;
    return 0;
  }
  Int_t n= (Int_t)(y+0.5);
  if(fabs(y-n)>1e-6) {
    cout << fName << "::roundBin: rounding non-integer bin contents: " << y << endl;
  }
  return n;
}

void RooHist::addBin(Axis_t binCenter, Int_t n) {
  // Add a bin to this histogram with the specified integer bin contents
  // and using an error bar calculated with Poisson statistics.

  Int_t index= GetN();
//    Double_t ym= RooMath::PoissonError(n,RooMath::NegativeError,_nSigma);
//    Double_t yp= RooMath::PoissonError(n,RooMath::PositiveError,_nSigma);
  Double_t ym= sqrt(n), yp= ym;
  SetPoint(index,binCenter,n);
  SetPointError(index,0,0,ym,yp);
  updateYAxisLimits(n+yp);
  updateYAxisLimits(n-ym);
}

void RooHist::addAsymmetryBin(Axis_t binCenter, Int_t n1, Int_t n2) {
  // Add a bin to this histogram with the value (n1-n2)/(n1+n2)
  // using an error bar calculated with Binomial statistics.

}

RooHist::~RooHist() { }

void RooHist::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this histogram to the specified output stream.
  //
  //   Standard: number of entries
  //      Shape: error CL and maximum value
  //    Verbose: print our bin contents and errors

  oneLinePrint(os,*this);
  RooPlotable::printToStream(os,opt,indent);
  if(opt >= Standard) {
    os << indent << "--- RooHist ---" << endl;
    Int_t n= GetN();
    os << indent << "  Contains " << n << " bins" << endl;
    if(opt >= Shape) {
      os << indent << "  Errors calculated at" << _nSigma << "-sigma CL" << endl;
      if(opt >= Verbose) {
	os << indent << "  Bin Contents:" << endl;
	for(Int_t i= 0; i < n; i++) {
	  os << indent << setw(3) << i << ") x= " <<  fX[i];
	  if(fEXhigh[i] > 0 || fEXlow[i] > 0) {
	    os << " +" << fEXhigh[i] << " -" << fEXlow[i];
	  }
	  os << " , y = " << fY[i] << " +" << fEYhigh[i] << " -" << fEYlow[i] << endl;
	}
      }
    }
  }
}
