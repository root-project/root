/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooHist.cc,v 1.8 2001/08/03 18:11:34 verkerke Exp $
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

// #include "BaBar/BaBar.hh"

#include "RooFitCore/RooHist.hh"
//#include "RooFitTools/RooMath.hh"

#include "TH1.h"
#include <iostream.h>
#include <iomanip.h>
#include <math.h>

ClassImp(RooHist)

static const char rcsid[] =
"$Id: RooHist.cc,v 1.8 2001/08/03 18:11:34 verkerke Exp $";

RooHist::RooHist(Double_t nominalBinWidth, Double_t nSigma) :
  TGraphAsymmErrors(), _nominalBinWidth(nominalBinWidth), _nSigma(nSigma)
{
  // Create an empty histogram that can be filled with the addBin()
  // and addAsymmetryBin() methods. Use the optional parameter to
  // specify the confidence level in units of sigma to use for
  // calculating error bars. The nominal bin width specifies the
  // default used by addBin(), and is used to set the relative
  // normalization of bins with different widths.

  initialize();
}

RooHist::RooHist(const TH1 &data, Double_t nominalBinWidth, Double_t nSigma) :
  TGraphAsymmErrors(), _nominalBinWidth(nominalBinWidth), _nSigma(nSigma)
{
  // Create a histogram from the contents of the specified TH1 object
  // which may have fixed or variable bin widths. Error bars are
  // calculated using Poisson statistics. Prints a warning and rounds
  // any bins with non-integer contents. Use the optional parameter to
  // specify the confidence level in units of sigma to use for
  // calculating error bars. The nominal bin width specifies the
  // default used by addBin(), and is used to set the relative
  // normalization of bins with different widths. If not set, the
  // nominal bin width is calculated as range/nbins.

  initialize();
  // copy the input histogram's name and title
  SetName(data.GetName());
  SetTitle(data.GetTitle());
  // calculate our nominal bin width if necessary
  if(_nominalBinWidth == 0) {
    const TAxis *axis= ((TH1&)data).GetXaxis();
    if(axis->GetNbins() > 0) _nominalBinWidth= (axis->GetXmax() - axis->GetXmin())/axis->GetNbins();
  }
  // TH1::GetYaxis() is not const (why!?)
  setYAxisLabel(const_cast<TH1&>(data).GetYaxis()->GetTitle());
  // initialize our contents from the input histogram's contents
  Int_t nbin= data.GetNbinsX();
  for(Int_t bin= 1; bin <= nbin; bin++) {
    Axis_t x= data.GetBinCenter(bin);
    Stat_t y= data.GetBinContent(bin);
    addBin(x,roundBin(y),data.GetBinWidth(bin));
  }
  // add over/underflow bins to our event count
  _entries+= data.GetBinContent(0) + data.GetBinContent(nbin+1);
}

void RooHist::initialize() {
  // Perform common initialization for all constructors.

  SetMarkerStyle(8);
  _entries= 0;
}

Double_t RooHist::getFitRangeNEvt() const {
  return _entries ;
}

Double_t RooHist::getFitRangeBinW() const {
  return _nominalBinWidth ;
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

void RooHist::addBin(Axis_t binCenter, Int_t n, Double_t binWidth) {
  // Add a bin to this histogram with the specified integer bin contents
  // and using an error bar calculated with Poisson statistics. The bin width
  // is used to set the relative scale of bins with different widths.

  Double_t scale= 1;
  if(binWidth > 0) scale= _nominalBinWidth/binWidth;
  _entries+= n;
  Int_t index= GetN();
//    Double_t ym= RooPoisson::NegativeError(n,_nSigma);
//    Double_t yp= RooMath::PoissonError(n,RooMath::PositiveError,_nSigma);
  Double_t ym= sqrt(n), yp= ym, dx= 0.5*binWidth;
  SetPoint(index,binCenter,n);
  SetPointError(index,dx,dx,scale*ym,scale*yp);
  updateYAxisLimits(scale*(n+yp));
  updateYAxisLimits(scale*(n-ym));
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
