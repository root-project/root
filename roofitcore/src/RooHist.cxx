/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooHist.cc,v 1.35 2006/12/08 15:50:40 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

// -- CLASS DESCRIPTION [PLOT] --
// A RooHist is a graphical representation of binned data based on the
// TGraphAsymmErrors class. Error bars are calculated using either Poisson
// or Binomial statistics.

#include "RooFit.h"

#include "RooHist.h"
#include "RooHist.h"
#include "RooHistError.h"
#include "RooCurve.h"

#include "TH1.h"
#include "Riostream.h"
#include <iomanip>
#include <math.h>

ClassImp(RooHist)

  RooHist::RooHist(Double_t nominalBinWidth, Double_t nSigma, Double_t /*xErrorFrac*/) :
    TGraphAsymmErrors(), _nominalBinWidth(nominalBinWidth), _nSigma(nSigma), _rawEntries(-1)
{
  // Create an empty histogram that can be filled with the addBin()
  // and addAsymmetryBin() methods. Use the optional parameter to
  // specify the confidence level in units of sigma to use for
  // calculating error bars. The nominal bin width specifies the
  // default used by addBin(), and is used to set the relative
  // normalization of bins with different widths.

  initialize();
}

RooHist::RooHist(const TH1 &data, Double_t nominalBinWidth, Double_t nSigma, RooAbsData::ErrorType etype, Double_t xErrorFrac) :
  TGraphAsymmErrors(), _nominalBinWidth(nominalBinWidth), _nSigma(nSigma), _rawEntries(-1)
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
    Stat_t dy = data.GetBinError(bin) ;
    if (etype==RooAbsData::Poisson) {
      addBin(x,roundBin(y),data.GetBinWidth(bin),xErrorFrac);
    } else {
      addBinWithError(x,y,dy,dy,data.GetBinWidth(bin),xErrorFrac);
    }
  }
  // add over/underflow bins to our event count
  _entries+= data.GetBinContent(0) + data.GetBinContent(nbin+1);
}



RooHist::RooHist(const TH1 &data1, const TH1 &data2, Double_t nominalBinWidth, Double_t nSigma, Double_t xErrorFrac) :
  TGraphAsymmErrors(), _nominalBinWidth(nominalBinWidth), _nSigma(nSigma), _rawEntries(-1)
{
  // Create a histogram from the asymmetry between the specified TH1 objects
  // which may have fixed or variable bin widths, but which must both have
  // the same binning. The asymmetry is calculated as (1-2)/(1+2). Error bars are
  // calculated using Binomial statistics. Prints a warning and rounds
  // any bins with non-integer contents. Use the optional parameter to
  // specify the confidence level in units of sigma to use for
  // calculating error bars. The nominal bin width specifies the
  // default used by addAsymmetryBin(), and is used to set the relative
  // normalization of bins with different widths. If not set, the
  // nominal bin width is calculated as range/nbins.

  initialize();
  // copy the first input histogram's name and title
  SetName(data1.GetName());
  SetTitle(data1.GetTitle());
  // calculate our nominal bin width if necessary
  if(_nominalBinWidth == 0) {
    const TAxis *axis= ((TH1&)data1).GetXaxis();
    if(axis->GetNbins() > 0) _nominalBinWidth= (axis->GetXmax() - axis->GetXmin())/axis->GetNbins();
  }
  setYAxisLabel(Form("Asymmetry (%s - %s)/(%s + %s)",
		     data1.GetName(),data2.GetName(),data1.GetName(),data2.GetName()));
  // initialize our contents from the input histogram contents
  Int_t nbin= data1.GetNbinsX();
  if(data2.GetNbinsX() != nbin) {
    cout << "RooHist::RooHist: histograms have different number of bins" << endl;
    return;
  }
  for(Int_t bin= 1; bin <= nbin; bin++) {
    Axis_t x= data1.GetBinCenter(bin);
    if(fabs(data2.GetBinCenter(bin)-x)>1e-10) {
      cout << "RooHist::RooHist: histograms have different centers for bin " << bin << endl;
    }
    Stat_t y1= data1.GetBinContent(bin);
    Stat_t y2= data2.GetBinContent(bin);
    addAsymmetryBin(x,roundBin(y1),roundBin(y2),data1.GetBinWidth(bin),xErrorFrac);
  }
  // we do not have a meaningful number of entries
  _entries= -1;
}


RooHist::RooHist(const RooHist& hist1, const RooHist& hist2, Double_t wgt1, Double_t wgt2, RooAbsData::ErrorType etype, Double_t xErrorFrac) : _rawEntries(-1){
  // Create histogram as sum of two existing histograms. If Poisson errors are selected the histograms are
  // added and Poisson confidence intervals are calculated for the summed content. If wgt1 and wgt2 are not
  // 1 in this mode, a warning message is printed. If SumW2 errors are selectd the histograms are added
  // and the histograms errors are added in quadrature, taking the weights into account.

  // Initialize the histogram
  initialize() ;
     
  // Copy all non-content properties from hist1
  SetName(hist1.GetName()) ;
  SetTitle(hist1.GetTitle()) ;  
  _nominalBinWidth=hist1._nominalBinWidth ;
  _nSigma=hist1._nSigma ;
  setYAxisLabel(hist1.getYAxisLabel()) ;

  if (!hist1.hasIdenticalBinning(hist2)) {
    cout << "RooHist::RooHist input histograms have incompatible binning, combined histogram will remain empty" << endl ;
    return ;
  }

  if (etype==RooAbsData::Poisson) {
    // Add histograms with Poisson errors

    // Issue warning if weights are not 1
    if (wgt1!=1.0 || wgt2 != 1.0) {
      cout << "RooHist::RooHist: WARNING: Poisson errors of weighted sum of two histograms is not well defined! " << endl
	   << "                  Summed histogram bins will rounded to nearest integer for Poisson confidence interval calculation" << endl ;
    }

    // Add histograms, calculate Poisson confidence interval on sum value
    Int_t i,n=hist1.GetN() ;
    for(i=0 ; i<n ; i++) {
      Double_t x1,y1,x2,y2,dx1 ;
#if ROOT_VERSION_CODE >= ROOT_VERSION(4,0,1)
      hist1.GetPoint(i,x1,y1) ;
#else
      const_cast<RooHist&>(hist1).GetPoint(i,x1,y1) ;
#endif
      dx1 = hist1.GetErrorX(i) ;
#if ROOT_VERSION_CODE >= ROOT_VERSION(4,0,1)
      hist2.GetPoint(i,x2,y2) ;
#else
      const_cast<RooHist&>(hist2).GetPoint(i,x2,y2) ;
#endif
      addBin(x1,roundBin(wgt1*y1+wgt2*y2),2*dx1/xErrorFrac,xErrorFrac) ;
    }    

  } else {
    // Add histograms with SumW2 errors

    // Add histograms, calculate combined sum-of-weights error
    Int_t i,n=hist1.GetN() ;
    for(i=0 ; i<n ; i++) {
      Double_t x1,y1,x2,y2,dx1,dy1,dy2 ;
#if ROOT_VERSION_CODE >= ROOT_VERSION(4,0,1)
      hist1.GetPoint(i,x1,y1) ;
#else
      const_cast<RooHist&>(hist1).GetPoint(i,x1,y1) ;
#endif
      dx1 = hist1.GetErrorX(i) ;
      dy1 = hist1.GetErrorY(i) ;
      dy2 = hist2.GetErrorY(i) ;
#if ROOT_VERSION_CODE >= ROOT_VERSION(4,0,1)
      hist2.GetPoint(i,x2,y2) ;
#else
      const_cast<RooHist&>(hist2).GetPoint(i,x2,y2) ;
#endif
      Double_t dy = sqrt(wgt1*wgt1*dy1*dy1+wgt2*wgt2*dy2*dy2) ;
      addBinWithError(x1,wgt1*y1+wgt2*y2,dy,dy,2*dx1/xErrorFrac,xErrorFrac) ;
    }       
  }

}

void RooHist::initialize() {
  // Perform common initialization for all constructors.

  SetMarkerStyle(8);
  _entries= 0;
}

Double_t RooHist::getFitRangeNEvt() const {
  return (_rawEntries==-1 ? _entries : _rawEntries) ;
}

Double_t RooHist::getFitRangeNEvt(Double_t xlo, Double_t xhi) const 
{
  // Calculate integral of histogram in given range 
  Double_t sum(0) ;
  for (int i=0 ; i<GetN() ; i++) {
    Double_t x,y ;

#if ROOT_VERSION_CODE >= ROOT_VERSION(4,0,1)
    GetPoint(i,x,y) ;
#else
    const_cast<RooHist*>(this)->GetPoint(i,x,y) ;
#endif

    if (x>=xlo && x<=xhi) {
      sum += y ;
    }
  }
  
  if (_rawEntries!=-1) {
    cout << "RooHist::getFitRangeNEvt() WARNING: Number of normalization events associated to histogram is not equal to number of events in histogram" << endl
	 << "                           due cut made in RooAbsData::plotOn() call. Automatic normalization over sub-range of plot variable assumes"    << endl
         << "                           that the effect of that cut is uniform across the plot, which may be an incorrect assumption. To be sure of"   << endl 
         << "                           correct normalization explicit pass normalization information to RooAbsPdf::plotOn() call using Normalization()" << endl ;
    sum *= _rawEntries / _entries ;
  }

  return sum ;
}


Double_t RooHist::getFitRangeBinW() const {
  return _nominalBinWidth ;
}


Int_t RooHist::roundBin(Double_t y) {
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

void RooHist::addBin(Axis_t binCenter, Int_t n, Double_t binWidth, Double_t xErrorFrac) {
  // Add a bin to this histogram with the specified integer bin contents
  // and using an error bar calculated with Poisson statistics. The bin width
  // is used to set the relative scale of bins with different widths.

  Double_t scale= 1;
  if(binWidth > 0) {
    scale= _nominalBinWidth/binWidth;
  }  
  _entries+= n;
  Int_t index= GetN();

  // calculate Poisson errors for this bin
  Double_t ym,yp,dx(0.5*binWidth);
  if(!RooHistError::instance().getPoissonInterval(n,ym,yp,_nSigma)) {
    cout << "RooHist::addBin: unable to add bin with " << n << " events" << endl;
    return;
  }

  SetPoint(index,binCenter,n*scale);
  SetPointError(index,dx*xErrorFrac,dx*xErrorFrac,scale*(n-ym),scale*(yp-n));
  updateYAxisLimits(scale*yp);
  updateYAxisLimits(scale*ym);
}



void RooHist::addBinWithError(Axis_t binCenter, Double_t n, Double_t elow, Double_t ehigh, Double_t binWidth, Double_t xErrorFrac) 
{
  // Add a bin to this histogram with the specified bin contents
  // and error. The bin width is used to set the relative scale of 
  // bins with different widths.

  Double_t scale= 1;
  if(binWidth > 0) {
    scale= _nominalBinWidth/binWidth;
  }  
  _entries+= n;
  Int_t index= GetN();

  Double_t dx(0.5*binWidth) ;
  SetPoint(index,binCenter,n*scale);
  SetPointError(index,dx*xErrorFrac,dx*xErrorFrac,elow*scale,ehigh*scale);
  updateYAxisLimits(scale*(n-elow));
  updateYAxisLimits(scale*(n+ehigh));
}






void RooHist::addAsymmetryBin(Axis_t binCenter, Int_t n1, Int_t n2, Double_t binWidth, Double_t xErrorFrac) {
  // Add a bin to this histogram with the value (n1-n2)/(n1+n2)
  // using an error bar calculated with Binomial statistics.

  Double_t scale= 1;
  if(binWidth > 0) scale= _nominalBinWidth/binWidth;
  Int_t index= GetN();

  // calculate Binomial errors for this bin
  Double_t ym,yp,dx(0.5*binWidth);
  if(!RooHistError::instance().getBinomialInterval(n1,n2,ym,yp,_nSigma)) {
    cout << "RooHist::addAsymmetryBin: unable to calculate binomial error for bin with " << n1 << "," << n2 << " events" << endl;
    return;
  }

  Double_t a= (Double_t)(n1-n2)/(n1+n2);
  SetPoint(index,binCenter,a);
  SetPointError(index,dx*xErrorFrac,dx*xErrorFrac,(a-ym),(yp-a));
  updateYAxisLimits(scale*yp);
  updateYAxisLimits(scale*ym);
}


RooHist::~RooHist() { }


Bool_t RooHist::hasIdenticalBinning(const RooHist& other) const 
{
  // First check if number of bins is the same
  if (GetN() != other.GetN()) {
    return kFALSE ;
  }

  // Next require that all bin centers are the same
  Int_t i ;
  for (i=0 ; i<GetN() ; i++) {
    Double_t x1,x2,y1,y2 ;
    
#if ROOT_VERSION_CODE >= ROOT_VERSION(4,0,1)
    GetPoint(i,x1,y1) ;
    other.GetPoint(i,x2,y2) ;
#else
    const_cast<RooHist&>(*this).GetPoint(i,x1,y1) ;
    const_cast<RooHist&>(other).GetPoint(i,x2,y2) ;
#endif

    if (fabs(x1-x2)>1e-10) {
      return kFALSE ;
    }

  }

  return kTRUE ;
}


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



RooHist* RooHist::makeResidHist(const RooCurve& curve,bool normalize) const {
  // Make histogram of (normalized) residuals w.r.t to given curve

  // Copy all non-content properties from hist1
  RooHist* hist = new RooHist(_nominalBinWidth) ;
  hist->SetName(Form(normalize?"pull_%s_s":"resid_%s_s",GetName(),curve.GetName())) ;
  hist->SetTitle(Form(normalize?"Pull of %s and %s":"Residual of %s and %s",GetTitle(),curve.GetTitle())) ;  

  // Determine range of curve 
  Double_t xstart,xstop,y ;
#if ROOT_VERSION_CODE >= ROOT_VERSION(4,0,1)
  curve.GetPoint(0,xstart,y) ;
  curve.GetPoint(curve.GetN()-1,xstop,y) ;
#else
  const_cast<RooCurve&>(curve).GetPoint(0,xstart,y) ;
  const_cast<RooCurve&>(curve).GetPoint(curve.GetN()-1,xstop,y) ;
#endif
  
  // Add histograms, calculate Poisson confidence interval on sum value
  for(Int_t i=0 ; i<GetN() ; i++) {    
    Double_t x,point;
#if ROOT_VERSION_CODE >= ROOT_VERSION(4,0,1)
    GetPoint(i,x,point) ;
#else
    const_cast<RooHist&>(*this).GetPoint(i,x,point) ;
#endif

    // Only calculate pull for bins inside curve range
    if (x<xstart || x>xstop) continue ;

    Double_t y = point - curve.interpolate(x) ;
    Double_t dyl = GetErrorYlow(i) ;
    Double_t dyh = GetErrorYhigh(i) ;
    if (normalize) {
        Double_t norm = (y>0?dyh:dyl);
	if (norm==0.) {
	  cout << "RooHist::makeResisHist(" << GetName() << ") WARNING: point " << i << " has zero error, setting residual to zero" << endl ;
	  y=0 ;
	  dyh=0 ;
	  dyl=0 ;
	} else {
	  y   /= norm;
	  dyh /= norm;
	  dyl /= norm;
	}
    }
    hist->addBinWithError(x,y,dyl,dyh);
  }
  return hist ;
}
