/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

/**
\file RooHist.cxx
\class RooHist
\ingroup Roofitcore

Graphical representation of binned data based on the
TGraphAsymmErrors class. Error bars are calculated using either Poisson
or Binomial statistics. A RooHist is used to represent histograms in
a RooPlot.
**/

#include "RooHist.h"

#include "RooAbsRealLValue.h"
#include "RooHistError.h"
#include "RooCurve.h"
#include "RooMsgService.h"
#include "RooProduct.h"
#include "RooConstVar.h"

#include "TH1.h"
#include "Riostream.h"
#include <iomanip>



////////////////////////////////////////////////////////////////////////////////
/// Create an empty histogram that can be filled with the addBin()
/// and addAsymmetryBin() methods. Use the optional parameter to
/// specify the confidence level in units of sigma to use for
/// calculating error bars. The nominal bin width specifies the
/// default used by addBin(), and is used to set the relative
/// normalization of bins with different widths.

RooHist::RooHist(double nominalBinWidth, double nSigma, double /*xErrorFrac*/, double /*scaleFactor*/)
   : _nominalBinWidth(nominalBinWidth), _nSigma(nSigma), _rawEntries(-1)
{
  initialize();
}


////////////////////////////////////////////////////////////////////////////////
/// Create a histogram from the contents of the specified TH1 object
/// which may have fixed or variable bin widths. Error bars are
/// calculated using Poisson statistics. Prints a warning and rounds
/// any bins with non-integer contents. Use the optional parameter to
/// specify the confidence level in units of sigma to use for
/// calculating error bars. The nominal bin width specifies the
/// default used by addBin(), and is used to set the relative
/// normalization of bins with different widths. If not set, the
/// nominal bin width is calculated as range/nbins.

RooHist::RooHist(const TH1 &data, double nominalBinWidth, double nSigma, RooAbsData::ErrorType etype, double xErrorFrac,
                 bool correctForBinWidth, double scaleFactor)
   : _nominalBinWidth(nominalBinWidth), _nSigma(nSigma), _rawEntries(-1)
{
  if(etype == RooAbsData::Poisson && correctForBinWidth == false) {
    throw std::invalid_argument(
            "To ensure consistent behavior prior releases, it's not possible to create a RooHist from a TH1 with no bin width correction when using Poisson errors.");
  }

  initialize();
  // copy the input histogram's name and title
  SetName(data.GetName());
  SetTitle(data.GetTitle());
  // calculate our nominal bin width if necessary
  if(_nominalBinWidth == 0) {
    const TAxis *axis= ((TH1&)data).GetXaxis();
    if(axis->GetNbins() > 0) _nominalBinWidth= (axis->GetXmax() - axis->GetXmin())/axis->GetNbins();
  }
  setYAxisLabel(data.GetYaxis()->GetTitle());

  // initialize our contents from the input histogram's contents
  Int_t nbin= data.GetNbinsX();
  for(Int_t bin= 1; bin <= nbin; bin++) {
    Axis_t x= data.GetBinCenter(bin);
    Stat_t y= data.GetBinContent(bin);
    Stat_t dy = data.GetBinError(bin) ;
    if (etype==RooAbsData::Poisson) {
      addBin(x,y,data.GetBinWidth(bin),xErrorFrac,scaleFactor);
    } else if (etype==RooAbsData::SumW2) {
      addBinWithError(x,y,dy,dy,data.GetBinWidth(bin),xErrorFrac,correctForBinWidth,scaleFactor);
    } else {
      addBinWithError(x,y,0,0,data.GetBinWidth(bin),xErrorFrac,correctForBinWidth,scaleFactor);
    }
  }
  // add over/underflow bins to our event count
  _entries+= data.GetBinContent(0) + data.GetBinContent(nbin+1);
}



////////////////////////////////////////////////////////////////////////////////
/// Create a histogram from the asymmetry between the specified TH1 objects
/// which may have fixed or variable bin widths, but which must both have
/// the same binning. The asymmetry is calculated as (1-2)/(1+2). Error bars are
/// calculated using Binomial statistics. Prints a warning and rounds
/// any bins with non-integer contents. Use the optional parameter to
/// specify the confidence level in units of sigma to use for
/// calculating error bars. The nominal bin width specifies the
/// default used by addAsymmetryBin(), and is used to set the relative
/// normalization of bins with different widths. If not set, the
/// nominal bin width is calculated as range/nbins.

RooHist::RooHist(const TH1 &data1, const TH1 &data2, double nominalBinWidth, double nSigma, RooAbsData::ErrorType etype,
                 double xErrorFrac, bool efficiency, double scaleFactor)
   : _nominalBinWidth(nominalBinWidth), _nSigma(nSigma), _rawEntries(-1)
{
  initialize();
  // copy the first input histogram's name and title
  SetName(data1.GetName());
  SetTitle(data1.GetTitle());
  // calculate our nominal bin width if necessary
  if(_nominalBinWidth == 0) {
    const TAxis *axis= data1.GetXaxis();
    if(axis->GetNbins() > 0) _nominalBinWidth= (axis->GetXmax() - axis->GetXmin())/axis->GetNbins();
  }

  if (!efficiency) {
    setYAxisLabel(Form("Asymmetry (%s - %s)/(%s + %s)",
           data1.GetName(),data2.GetName(),data1.GetName(),data2.GetName()));
  } else {
    setYAxisLabel(Form("Efficiency (%s)/(%s + %s)",
           data1.GetName(),data1.GetName(),data2.GetName()));
  }
  // initialize our contents from the input histogram contents
  Int_t nbin= data1.GetNbinsX();
  if(data2.GetNbinsX() != nbin) {
    coutE(InputArguments) << "RooHist::RooHist: histograms have different number of bins" << std::endl;
    return;
  }
  for(Int_t bin= 1; bin <= nbin; bin++) {
    Axis_t x= data1.GetBinCenter(bin);
    if(std::abs(data2.GetBinCenter(bin)-x)>1e-10) {
      coutW(InputArguments) << "RooHist::RooHist: histograms have different centers for bin " << bin << std::endl;
    }
    Stat_t y1= data1.GetBinContent(bin);
    Stat_t y2= data2.GetBinContent(bin);
    if (!efficiency) {

      if (etype==RooAbsData::Poisson) {
   addAsymmetryBin(x,roundBin(y1),roundBin(y2),data1.GetBinWidth(bin),xErrorFrac,scaleFactor);
      } else if (etype==RooAbsData::SumW2) {
   Stat_t dy1= data1.GetBinError(bin);
   Stat_t dy2= data2.GetBinError(bin);
   addAsymmetryBinWithError(x,y1,y2,dy1,dy2,data1.GetBinWidth(bin),xErrorFrac,scaleFactor);
      } else {
   addAsymmetryBinWithError(x,y1,y2,0,0,data1.GetBinWidth(bin),xErrorFrac,scaleFactor);
      }

    } else {

      if (etype==RooAbsData::Poisson) {
   addEfficiencyBin(x,roundBin(y1),roundBin(y2),data1.GetBinWidth(bin),xErrorFrac,scaleFactor);
      } else if (etype==RooAbsData::SumW2) {
   Stat_t dy1= data1.GetBinError(bin);
   Stat_t dy2= data2.GetBinError(bin);
   addEfficiencyBinWithError(x,y1,y2,dy1,dy2,data1.GetBinWidth(bin),xErrorFrac,scaleFactor);
      } else {
   addEfficiencyBinWithError(x,y1,y2,0,0,data1.GetBinWidth(bin),xErrorFrac,scaleFactor);
      }

    }

  }
  // we do not have a meaningful number of entries
  _entries= -1;
}



////////////////////////////////////////////////////////////////////////////////
/// Create histogram as sum of two existing histograms. If Poisson errors are selected the histograms are
/// added and Poisson confidence intervals are calculated for the summed content. If wgt1 and wgt2 are not
/// 1 in this mode, a warning message is printed. If SumW2 errors are selected the histograms are added
/// and the histograms errors are added in quadrature, taking the weights into account.

RooHist::RooHist(const RooHist &hist1, const RooHist &hist2, double wgt1, double wgt2, RooAbsData::ErrorType etype,
                 double xErrorFrac)
   : _nominalBinWidth(hist1._nominalBinWidth), _nSigma(hist1._nSigma), _rawEntries(-1)
{
  // Initialize the histogram
  initialize() ;

  // Copy all non-content properties from hist1
  SetName(hist1.GetName()) ;
  SetTitle(hist1.GetTitle()) ;

  setYAxisLabel(hist1.getYAxisLabel()) ;

  if (!hist1.hasIdenticalBinning(hist2)) {
    coutE(InputArguments) << "RooHist::RooHist input histograms have incompatible binning, combined histogram will remain empty" << std::endl ;
    return ;
  }

  if (etype==RooAbsData::Poisson) {
    // Add histograms with Poisson errors

    // Issue warning if weights are not 1
    if (wgt1!=1.0 || wgt2 != 1.0) {
      coutW(InputArguments) << "RooHist::RooHist: WARNING: Poisson errors of weighted sum of two histograms is not well defined! " << std::endl
             << "                  Summed histogram bins will rounded to nearest integer for Poisson confidence interval calculation" << std::endl ;
    }

    // Add histograms, calculate Poisson confidence interval on sum value
    Int_t i;
    Int_t n = hist1.GetN();
    for(i=0 ; i<n ; i++) {
      double x1;
      double y1;
      double x2;
      double y2;
      double dx1;
      hist1.GetPoint(i,x1,y1) ;
      dx1 = hist1.GetErrorX(i) ;
      hist2.GetPoint(i,x2,y2) ;
      addBin(x1,roundBin(wgt1*y1+wgt2*y2),2*dx1/xErrorFrac,xErrorFrac) ;
    }

  } else {
    // Add histograms with SumW2 errors

    // Add histograms, calculate combined sum-of-weights error
    Int_t i;
    Int_t n = hist1.GetN();
    for(i=0 ; i<n ; i++) {
      double x1;
      double y1;
      double x2;
      double y2;
      double dx1;
      double dy1;
      double dy2;
      hist1.GetPoint(i,x1,y1) ;
      dx1 = hist1.GetErrorX(i) ;
      dy1 = hist1.GetErrorY(i) ;
      dy2 = hist2.GetErrorY(i) ;
      hist2.GetPoint(i,x2,y2) ;
      double dy = sqrt(wgt1*wgt1*dy1*dy1+wgt2*wgt2*dy2*dy2) ;
      addBinWithError(x1,wgt1*y1+wgt2*y2,dy,dy,2*dx1/xErrorFrac,xErrorFrac) ;
    }
  }

}


////////////////////////////////////////////////////////////////////////////////
/// Create histogram from a pdf or function. Errors are computed based on the fit result provided.
///
/// This signature is intended for unfolding/deconvolution scenarios,
/// where a pdf is constructed as "data minus background" and is thus
/// intended to be displayed as "data" (or at least data-like).
/// Usage of this signature is triggered by the draw style "P" in RooAbsReal::plotOn.
///
/// More details.
/// \param[in] f The function to be plotted.
/// \param[in] x The variable on the x-axis
/// \param[in] xErrorFrac Size of the error in x as a fraction of the bin width
/// \param[in] scaleFactor arbitrary scaling of the y-values
/// \param[in] normVars variables over which to normalize
/// \param[in] fr fit result
RooHist::RooHist(const RooAbsReal &f, RooAbsRealLValue &x, double xErrorFrac, double scaleFactor,
                 const RooArgSet *normVars, const RooFitResult *fr)
   : _nSigma(1), _rawEntries(-1)
{
  // grab the function's name and title
  SetName(f.GetName());
  std::string title{f.GetTitle()};
  SetTitle(title.c_str());
  // append " ( [<funit> ][/ <xunit> ])" to our y-axis label if necessary
  if(0 != strlen(f.getUnit()) || 0 != strlen(x.getUnit())) {
    title += " ( ";
    if(0 != strlen(f.getUnit())) {
      title += f.getUnit();
      title += " ";
    }
    if(0 != strlen(x.getUnit())) {
      title += "/ ";
      title += x.getUnit();
      title += " ";
    }
    title += ")";
  }
  setYAxisLabel(title.c_str());

  RooProduct scaledFunc{"scaled_func", "scaled_func", {f, RooFit::RooConst(scaleFactor)}};
  std::unique_ptr<RooAbsFunc> funcPtr{scaledFunc.bindVars(x, normVars, true)};

  // calculate the points to add to our curve
  int xbins = x.numBins();
  RooArgSet nset;
  if(normVars) nset.add(*normVars);
  for(int i=0; i<xbins; ++i){
    double xval = x.getBinning().binCenter(i);
    double xwidth = x.getBinning().binWidth(i);
    Axis_t xval_ax = xval;
    double yval = (*funcPtr)(&xval);
    double yerr = std::sqrt(yval);
    if(fr) yerr = f.getPropagatedError(*fr,nset);
    addBinWithError(xval_ax,yval,yerr,yerr,xwidth,xErrorFrac,false,scaleFactor) ;
    _entries += yval;
  }
  _nominalBinWidth = 1.;
}


////////////////////////////////////////////////////////////////////////////////
/// Perform common initialization for all constructors.

void RooHist::initialize()
{
  SetMarkerStyle(8);
}


////////////////////////////////////////////////////////////////////////////////
/// Return the number of events of the dataset associated with this RooHist.
/// This is the number of events in the RooHist itself, unless a different
/// value was specified through setRawEntries()

double RooHist::getFitRangeNEvt() const
{
  return (_rawEntries==-1 ? _entries : _rawEntries) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate integral of histogram in given range

double RooHist::getFitRangeNEvt(double xlo, double xhi) const
{
  double sum(0) ;
  for (int i=0 ; i<GetN() ; i++) {
    double x;
    double y;

    GetPoint(i,x,y) ;

    if (x>=xlo && x<=xhi) {
      // We have to use the original weights of the histogram, because the
      // scaled points have nothing to do anymore with event weights in the
      // case of non-uniform binning. For backwards compatibility with the
      // RooHist version 1, we first need to check if the `_originalWeights`
      // member is filled.
      sum += _originalWeights.empty() ? y : _originalWeights[i];
    }
  }

  if (_rawEntries!=-1) {
    coutW(Plotting) << "RooHist::getFitRangeNEvt() WARNING: The number of normalisation events associated to histogram " << GetName() << " is not equal to number of events in this histogram."
          << "\n\t\t This is due a cut being applied while plotting the data. Automatic normalisation over a sub-range of a plot variable assumes"
          << "\n\t\t that the effect of that cut is uniform across the plot, which may be an incorrect assumption. To obtain a correct normalisation, it needs to be passed explicitly:"
          << "\n\t\t\t data->plotOn(frame01,CutRange(\"SB1\"));"
          << "\n\t\t\t const double nData = data->sumEntries(\"\", \"SB1\"); //or the cut string such as sumEntries(\"x > 0.\");"
          << "\n\t\t\t model.plotOn(frame01, RooFit::Normalization(nData, RooAbsReal::NumEvent), ProjectionRange(\"SB1\"));" << std::endl ;
    sum *= _rawEntries / _entries ;
  }

  return sum ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return the nearest positive integer to the input value
/// and print a warning if an adjustment is required.

Int_t RooHist::roundBin(double y)
{
  if(y < 0) {
    coutW(Plotting) << fName << "::roundBin: rounding negative bin contents to zero: " << y << std::endl;
    return 0;
  }
  Int_t n= (Int_t)(y+0.5);
  if(std::abs(y-n)>1e-6) {
    coutW(Plotting) << fName << "::roundBin: rounding non-integer bin contents: " << y << std::endl;
  }
  return n;
}


void RooHist::addPoint(Axis_t binCenter, double y, double yscale, double exlow, double exhigh, double eylow, double eyhigh)
{
  const int index = GetN();
  SetPoint(index, binCenter, y*yscale);

  // If the scale is negative, the low and high errors must be swapped
  if(std::abs(yscale) < 0) {
    std::swap(eylow, eyhigh);
  }

  SetPointError(index, exlow, exhigh, std::abs(yscale) * eylow, std::abs(yscale) * eyhigh);

  updateYAxisLimits(yscale * (y - eylow));
  updateYAxisLimits(yscale * (y + eyhigh));

  // We also track the original weights of the histogram, because if we only
  // have info on the scaled points it's not possible anymore to compute the
  // number of events in a subrange of the RooHist.
  _originalWeights.resize(index + 1);
  _originalWeights[index] = y;
}


////////////////////////////////////////////////////////////////////////////////
/// Add a bin to this histogram with the specified integer bin contents
/// and using an error bar calculated with Poisson statistics. The bin width
/// is used to set the relative scale of bins with different widths.

void RooHist::addBin(Axis_t binCenter, double n, double binWidth, double xErrorFrac, double scaleFactor)
{
  if (n<0) {
    coutW(Plotting) << "RooHist::addBin(" << GetName() << ") WARNING: negative entry set to zero when Poisson error bars are requested" << std::endl ;
  }

  double scale= 1;
  if(binWidth > 0) {
    scale= _nominalBinWidth/binWidth;
  }
  _entries+= n;

  // calculate Poisson errors for this bin
  double ym;
  double yp;
  double dx(0.5 * binWidth);

  if (std::abs((double)((n-Int_t(n))>1e-5))) {
    // need interpolation
    double ym1(0);
    double yp1(0);
    double ym2(0);
    double yp2(0);
    Int_t n1 = Int_t(n) ;
    Int_t n2 = n1+1 ;
    if(!RooHistError::instance().getPoissonInterval(n1,ym1,yp1,_nSigma) ||
       !RooHistError::instance().getPoissonInterval(n2,ym2,yp2,_nSigma)) {
      coutE(Plotting) << "RooHist::addBin: unable to add bin with " << n << " events" << std::endl;
    }
    ym = ym1 + (n-n1)*(ym2-ym1) ;
    yp = yp1 + (n-n1)*(yp2-yp1) ;
    coutW(Plotting) << "RooHist::addBin(" << GetName()
          << ") WARNING: non-integer bin entry " << n << " with Poisson errors, interpolating between Poisson errors of adjacent integer" << std::endl ;
  } else {
  // integer case
  if(!RooHistError::instance().getPoissonInterval(Int_t(n),ym,yp,_nSigma)) {
      coutE(Plotting) << "RooHist::addBin: unable to add bin with " << n << " events" << std::endl;
      return;
    }
  }

  addPoint(binCenter,n, scale*scaleFactor,dx*xErrorFrac,dx*xErrorFrac, n-ym, yp-n);
}



////////////////////////////////////////////////////////////////////////////////
/// Add a bin to this histogram with the specified bin contents
/// and error. The bin width is used to set the relative scale of
/// bins with different widths.

void RooHist::addBinWithError(Axis_t binCenter, double n, double elow, double ehigh, double binWidth,
               double xErrorFrac, bool correctForBinWidth, double scaleFactor)
{
  double scale= 1;
  if(binWidth > 0 && correctForBinWidth) {
    scale= _nominalBinWidth/binWidth;
  }
  _entries+= n;

  double dx(0.5*binWidth) ;
  addPoint(binCenter,n, scale*scaleFactor,dx*xErrorFrac,dx*xErrorFrac, elow, ehigh);
}




////////////////////////////////////////////////////////////////////////////////
/// Add a bin to this histogram with the specified bin contents
/// and error. The bin width is used to set the relative scale of
/// bins with different widths.

void RooHist::addBinWithXYError(Axis_t binCenter, double n, double exlow, double exhigh, double eylow, double eyhigh,
            double scaleFactor)
{
  _entries+= n;

  addPoint(binCenter, n, scaleFactor,exlow,exhigh, eylow, eyhigh);
}





////////////////////////////////////////////////////////////////////////////////
/// Add a bin to this histogram with the value (n1-n2)/(n1+n2)
/// using an error bar calculated with Binomial statistics.

void RooHist::addAsymmetryBin(Axis_t binCenter, Int_t n1, Int_t n2, double binWidth, double xErrorFrac, double scaleFactor)
{
  // calculate Binomial errors for this bin
  double ym;
  double yp;
  double dx(0.5 * binWidth);
  if(!RooHistError::instance().getBinomialIntervalAsym(n1,n2,ym,yp,_nSigma)) {
    coutE(Plotting) << "RooHist::addAsymmetryBin: unable to calculate binomial error for bin with " << n1 << "," << n2 << " events" << std::endl;
    return;
  }

  const Int_t denominator = n1 + n2;
  double a = 0 == denominator ? 0. : (double)(n1 - n2) / (denominator);
  addPoint(binCenter, a, scaleFactor,dx*xErrorFrac,dx*xErrorFrac, a-ym, yp-a);
}



////////////////////////////////////////////////////////////////////////////////
/// Add a bin to this histogram with the value (n1-n2)/(n1+n2)
/// using an error bar calculated with Binomial statistics.

void RooHist::addAsymmetryBinWithError(Axis_t binCenter, double n1, double n2, double en1, double en2, double binWidth, double xErrorFrac, double scaleFactor)
{
  // calculate Binomial errors for this bin
  double ym;
  double yp;
  double dx(0.5 * binWidth);
  double a= (double)(n1-n2)/(n1+n2);

  double error = 2*sqrt( pow(en1,2)*pow(n2,2) + pow(en2,2)*pow(n1,2) ) / pow(n1+n2,2) ;
  ym=a-error ;
  yp=a+error ;

  addPoint(binCenter,a, scaleFactor, dx*xErrorFrac,dx*xErrorFrac, a-ym, yp-a);
}



////////////////////////////////////////////////////////////////////////////////
/// Add a bin to this histogram with the value n1/(n1+n2)
/// using an error bar calculated with Binomial statistics.

void RooHist::addEfficiencyBin(Axis_t binCenter, Int_t n1, Int_t n2, double binWidth, double xErrorFrac, double scaleFactor)
{
  double a= (double)(n1)/(n1+n2);

  // calculate Binomial errors for this bin
  double ym;
  double yp;
  double dx(0.5 * binWidth);
  if(!RooHistError::instance().getBinomialIntervalEff(n1,n2,ym,yp,_nSigma)) {
    coutE(Plotting) << "RooHist::addEfficiencyBin: unable to calculate binomial error for bin with " << n1 << "," << n2 << " events" << std::endl;
    return;
  }

  addPoint(binCenter,a, scaleFactor,dx*xErrorFrac,dx*xErrorFrac, a-ym, yp-a);
}



////////////////////////////////////////////////////////////////////////////////
/// Add a bin to this histogram with the value n1/(n1+n2)
/// using an error bar calculated with Binomial statistics.

void RooHist::addEfficiencyBinWithError(Axis_t binCenter, double n1, double n2, double en1, double en2, double binWidth, double xErrorFrac, double scaleFactor)
{
  double a= (double)(n1)/(n1+n2);

  double error = sqrt( pow(en1,2)*pow(n2,2) + pow(en2,2)*pow(n1,2) ) / pow(n1+n2,2) ;

  // calculate Binomial errors for this bin
  double ym;
  double yp;
  double dx(0.5 * binWidth);
  ym=a-error ;
  yp=a+error ;


  addPoint(binCenter,a, scaleFactor,dx*xErrorFrac,dx*xErrorFrac, a-ym, yp-a);
}


////////////////////////////////////////////////////////////////////////////////
/// Return true if binning of this RooHist is identical to that of 'other'

bool RooHist::hasIdenticalBinning(const RooHist& other) const
{
  // First check if number of bins is the same
  if (GetN() != other.GetN()) {
    return false ;
  }

  // Next require that all bin centers are the same
  Int_t i ;
  for (i=0 ; i<GetN() ; i++) {
    double x1;
    double x2;
    double y1;
    double y2;

    GetPoint(i,x1,y1) ;
    other.GetPoint(i,x2,y2) ;

    if (std::abs(x1-x2) > 1e-10 * _nominalBinWidth) {
      return false ;
    }

  }

  return true ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return true if contents of this RooHist is identical within given
/// relative tolerance to that of 'other'

bool RooHist::isIdentical(const RooHist& other, double tol, bool verbose) const
{
  // Make temporary TH1s output of RooHists to perform Kolmogorov test
  TH1::AddDirectory(false) ;
  TH1F h_self("h_self","h_self",GetN(),0,1) ;
  TH1F h_other("h_other","h_other",GetN(),0,1) ;
  TH1::AddDirectory(true) ;

  for (Int_t i=0 ; i<GetN() ; i++) {
    h_self.SetBinContent(i+1,GetY()[i]) ;
    h_other.SetBinContent(i+1,other.GetY()[i]) ;
  }

  double M = h_self.KolmogorovTest(&h_other,"M") ;
  if (M>tol) {
    double kprob = h_self.KolmogorovTest(&h_other) ;
    if(verbose) std::cout << "RooHist::isIdentical() tolerance exceeded M=" << M << " (tol=" << tol << "), corresponding prob = " << kprob << std::endl ;
    return false ;
  }

  return true ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print info about this histogram to the specified output stream.
///
///   Standard: number of entries
///      Shape: error CL and maximum value
///    Verbose: print our bin contents and errors

void RooHist::printMultiline(std::ostream& os, Int_t contents, bool verbose, TString indent) const
{
  RooPlotable::printMultiline(os,contents,verbose,indent);
  os << indent << "--- RooHist ---" << std::endl;
  Int_t n= GetN();
  os << indent << "  Contains " << n << " bins" << std::endl;
  if(verbose) {
    os << indent << "  Errors calculated at" << _nSigma << "-sigma CL" << std::endl;
    os << indent << "  Bin Contents:" << std::endl;
    for(Int_t i= 0; i < n; i++) {
      os << indent << std::setw(3) << i << ") x= " <<  fX[i];
      if(fEXhigh[i] > 0 || fEXlow[i] > 0) {
   os << " +" << fEXhigh[i] << " -" << fEXlow[i];
      }
      os << " , y = " << fY[i] << " +" << fEYhigh[i] << " -" << fEYlow[i] << std::endl;
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Print name of RooHist

void RooHist::printName(std::ostream& os) const
{
  os << GetName() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print title of RooHist

void RooHist::printTitle(std::ostream& os) const
{
  os << GetTitle() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print class name of RooHist

void RooHist::printClassName(std::ostream& os) const
{
  os << ClassName() ;
}


std::unique_ptr<RooHist> RooHist::createEmptyResidHist(const RooCurve& curve, bool normalize) const
{
  // Copy all non-content properties from hist1
  auto hist = std::make_unique<RooHist>(_nominalBinWidth) ;
  const std::string name = GetName() + std::string("_") + curve.GetName();
  const std::string title = GetTitle() + std::string(" and ") + curve.GetTitle();
  hist->SetName(((normalize ? "pull_" : "resid_") + name).c_str()) ;
  hist->SetTitle(((normalize ? "Pull of " : "Residual of ") + title).c_str()) ;

  return hist;
}


void RooHist::fillResidHist(RooHist & residHist, const RooCurve& curve,bool normalize, bool useAverage) const
{
  // Determine range of curve
  double xstart;
  double xstop;
  double y;
  curve.GetPoint(0,xstart,y) ;
  curve.GetPoint(curve.GetN()-1,xstop,y) ;

  // Add histograms, calculate Poisson confidence interval on sum value
  for(Int_t i=0 ; i<GetN() ; i++) {
    double x;
    double point;
    GetPoint(i,x,point) ;

    // Only calculate pull for bins inside curve range
    if (x<xstart || x>xstop) continue ;

    double yy ;
    if (useAverage) {
      double exl = GetErrorXlow(i);
      double exh = GetErrorXhigh(i) ;
      if (exl<=0 ) exl = GetErrorX(i);
      if (exh<=0 ) exh = GetErrorX(i);
      if (exl<=0 ) exl = 0.5*getNominalBinWidth();
      if (exh<=0 ) exh = 0.5*getNominalBinWidth();
      yy = point - curve.average(x-exl,x+exh) ;
    } else {
      yy = point - curve.interpolate(x) ;
    }

    double dyl = GetErrorYlow(i) ;
    double dyh = GetErrorYhigh(i) ;
    if (normalize) {
        double norm = (yy>0?dyl:dyh);
   if (norm==0.) {
     coutW(Plotting) << "RooHist::makeResisHist(" << GetName() << ") WARNING: point " << i << " has zero error, setting residual to zero" << std::endl;
     yy=0 ;
     dyh=0 ;
     dyl=0 ;
   } else {
     yy   /= norm;
     dyh /= norm;
     dyl /= norm;
   }
    }
    residHist.addBinWithError(x,yy,dyl,dyh);
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Create and return RooHist containing  residuals w.r.t to given curve.
/// If normalize is true, the residuals are normalized by the histogram
/// errors creating a RooHist with pull values

RooHist* RooHist::makeResidHist(const RooCurve& curve, bool normalize, bool useAverage) const
{
  RooHist* hist = createEmptyResidHist(curve, normalize).release();
  fillResidHist(*hist, curve, normalize, useAverage);
  return hist ;
}
