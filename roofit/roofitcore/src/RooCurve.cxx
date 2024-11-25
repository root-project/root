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
\file RooCurve.cxx
\class RooCurve
\ingroup Roofitcore

One-dimensional graphical representation of a real-valued function.
A curve is approximated by straight line segments with end points chosen to give
a "good" approximation to the true curve. The goodness of the approximation is
controlled by a precision and a resolution parameter.

A RooCurve derives from TGraph, so it can either be drawn as a line (default) or
as points:
```
RooPlot *p = y.plotOn(x.frame());
p->getAttMarker("curve_y")->SetMarkerStyle(20);
p->setDrawOptions("curve_y","PL");
p->Draw();
```

To retrieve a RooCurve from a RooPlot, use RooPlot::getCurve().
**/

#include "RooCurve.h"
#include "RooHist.h"
#include "RooAbsReal.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooRealBinding.h"
#include "RooMsgService.h"
#include "RooProduct.h"
#include "RooConstVar.h"

#include "Riostream.h"
#include "TMath.h"
#include "TAxis.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include "Math/Util.h"
#include <iomanip>
#include <deque>
#include <algorithm>

using std::endl, std::ostream, std::list, std::vector, std::cout, std::min;

ClassImp(RooCurve);

namespace {

// Helpers to manage points
struct Point {
   double x;
   double y;
};

inline Point getPoint(TGraph const &gr, int i)
{
   Point p;
   gr.GetPoint(i, p.x, p.y);
   return p;
}

} // namespace


////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

RooCurve::RooCurve()
{
  initialize();
}


////////////////////////////////////////////////////////////////////////////////
/// Create a 1-dim curve of the value of the specified real-valued expression
/// as a function of x. Use the optional precision parameter to control
/// how precisely the smooth curve is rasterized. Use the optional argument set
/// to specify how the expression should be normalized. Use the optional scale
/// factor to rescale the expression after normalization.
/// If shiftToZero is set, the entire curve is shifted down to make the lowest
/// point of the curve go through zero.
RooCurve::RooCurve(const RooAbsReal &f, RooAbsRealLValue &x, double xlo, double xhi, Int_t xbins, double scaleFactor,
                   const RooArgSet *normVars, double prec, double resolution, bool shiftToZero, WingMode wmode,
                   Int_t nEvalError, Int_t doEEVal, double eeVal, bool showProg)
   : _showProgress(showProg)
{
  // grab the function's name and title
  TString name(f.GetName());
  SetName(name.Data());
  TString title(f.GetTitle());
  SetTitle(title.Data());
  // append " ( [<funit> ][/ <xunit> ])" to our y-axis label if necessary
  if(0 != strlen(f.getUnit()) || 0 != strlen(x.getUnit())) {
    title.Append(" ( ");
    if(0 != strlen(f.getUnit())) {
      title.Append(f.getUnit());
      title.Append(" ");
    }
    if(0 != strlen(x.getUnit())) {
      title.Append("/ ");
      title.Append(x.getUnit());
      title.Append(" ");
    }
    title.Append(")");
  }
  setYAxisLabel(title.Data());

  RooProduct scaledFunc{"scaled_func", "scaled_func", {f, RooFit::RooConst(scaleFactor)}};
  std::unique_ptr<RooAbsFunc> funcPtr{scaledFunc.bindVars(x, normVars, true)};

  // calculate the points to add to our curve
  if(xbins > 0){
    // regular mode - use the sampling hint to decide where to evaluate the pdf
    std::unique_ptr<std::list<double>> hint{f.plotSamplingHint(x,xlo,xhi)};
    addPoints(*funcPtr,xlo,xhi,xbins+1,prec,resolution,wmode,nEvalError,doEEVal,eeVal,hint.get());
    if (_showProgress) {
      ccoutP(Plotting) << endl ;
    }
  } else {
    // if number of bins is set to <= 0, skip any interpolation and just evaluate the pdf at the bin centers
    // this is useful when plotting a pdf like a histogram
    int nBinsX = x.numBins();
    for(int i=0; i<nBinsX; ++i){
      double xval = x.getBinning().binCenter(i);
      addPoint(xval,(*funcPtr)(&xval)) ;
    }
  }
  initialize();

  if (shiftToZero) shiftCurveToZero() ;

  // Adjust limits
  for (int i=0 ; i<GetN() ; i++) {
    updateYAxisLimits(fY[i]);
  }
  this->Sort();
}



////////////////////////////////////////////////////////////////////////////////
/// Create a 1-dim curve of the value of the specified real-valued
/// expression as a function of x. Use the optional precision
/// parameter to control how precisely the smooth curve is
/// rasterized. If shiftToZero is set, the entire curve is shifted
/// down to make the lowest point in of the curve go through zero.

RooCurve::RooCurve(const char *name, const char *title, const RooAbsFunc &func,
         double xlo, double xhi, UInt_t minPoints, double prec, double resolution,
         bool shiftToZero, WingMode wmode, Int_t nEvalError, Int_t doEEVal, double eeVal)
{
  SetName(name);
  SetTitle(title);
  addPoints(func,xlo,xhi,minPoints+1,prec,resolution,wmode,nEvalError,doEEVal,eeVal);
  initialize();
  if (shiftToZero) shiftCurveToZero() ;

  // Adjust limits
  for (int i=0 ; i<GetN() ; i++) {
    updateYAxisLimits(fY[i]);
  }
  this->Sort();
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor of a curve as sum of two other curves.
/// \f[
/// C_\mathrm{sum} = \mathrm{scale1}*c1 + \mathrm{scale2}*c2
/// \f]
///
/// \param[in] name Name of the curve (to retrieve it from a plot)
/// \param[in] title Title (for plotting).
/// \param[in] c1 First curve.
/// \param[in] c2 Second curve.
/// \param[in] scale1 Scale y values for c1 by this factor.
/// \param[in] scale2 Scale y values for c2 by this factor.

RooCurve::RooCurve(const char* name, const char* title, const RooCurve& c1, const RooCurve& c2, double scale1, double scale2)
{
  initialize() ;
  SetName(name) ;
  SetTitle(title) ;

  // Make deque of points in X
  std::deque<double> pointList ;

  // Add X points of C1
  Int_t n1 = c1.GetN();
  for (int i1=0 ; i1<n1 ; i1++) {
    pointList.push_back(c1.GetPointX(i1));
  }

  // Add X points of C2
  Int_t n2 = c2.GetN();
  for (int i2=0 ; i2<n2 ; i2++) {
    pointList.push_back(c2.GetPointX(i2));
  }

  // Sort X points
  std::sort(pointList.begin(),pointList.end()) ;

  // Loop over X points
  double last(-RooNumber::infinity()) ;
  for (auto point : pointList) {

    if ((point-last)>1e-10) {
      // Add OR of points to new curve, skipping duplicate points within tolerance
      addPoint(point,scale1*c1.interpolate(point)+scale2*c2.interpolate(point)) ;
    }
    last = point ;
  }

  this->Sort();
}


RooCurve::~RooCurve() = default;


////////////////////////////////////////////////////////////////////////////////
/// Perform initialization that is common to all curves

void RooCurve::initialize()
{
  // set default line width in pixels
  SetLineWidth(3);
  // set default line color
  SetLineColor(kBlue);
}



////////////////////////////////////////////////////////////////////////////////
/// Find lowest point in curve and move all points in curve so that
/// lowest point will go exactly through zero

void RooCurve::shiftCurveToZero()
{
   double minVal = std::numeric_limits<double>::infinity();
   double maxVal = -std::numeric_limits<double>::infinity();

   // First iteration, find current lowest point
   for (int i = 1; i < GetN() - 1; i++) {
      double y = GetPointY(i);
      minVal = std::min(y, minVal);
      maxVal = std::max(y, maxVal);
   }

   // Second iteration, lower all points by minVal
   for (int i = 1; i < GetN() - 1; i++) {
      Point point = getPoint(*this, i);
      SetPoint(i, point.x, point.y - minVal);
   }

   setYAxisLimits(0, maxVal - minVal);
}



////////////////////////////////////////////////////////////////////////////////
/// Add points calculated with the specified function, over the range (xlo,xhi).
/// Add at least minPoints equally spaced points, and add sufficient points so that
/// the maximum deviation from the final straight-line segments is prec*(ymax-ymin),
/// down to a minimum horizontal spacing of resolution*(xhi-xlo).

void RooCurve::addPoints(const RooAbsFunc &func, double xlo, double xhi,
          Int_t minPoints, double prec, double resolution, WingMode wmode,
          Int_t numee, bool doEEVal, double eeVal, list<double>* samplingHint)
{
  // check the inputs
  if(!func.isValid()) {
    coutE(InputArguments) << fName << "::addPoints: input function is not valid" << endl;
    return;
  }
  if(minPoints <= 0 || xhi <= xlo) {
    coutE(InputArguments) << fName << "::addPoints: bad input (nothing added)" << endl;
    return;
  }

  // Perform a coarse scan of the function to estimate its y range.
  // Save the results so we do not have to re-evaluate at the scan points.

  // Adjust minimum number of points to external sampling hint if used
  if (samplingHint) {
    minPoints = samplingHint->size() ;
  }

  double dx= (xhi-xlo)/(minPoints-1.);
  const double epsilon = (xhi - xlo) * relativeXEpsilon();
  std::vector<double> yval(minPoints);

  // Get list of initial x values. If function provides sampling hint use that,
  // otherwise use default binning of frame
  std::vector<double> xval;
  if (!samplingHint) {
    for(int step= 0; step < minPoints; step++) {
      xval.push_back(xlo + step*dx) ;
    }
  } else {
    std::copy(samplingHint->begin(), samplingHint->end(), std::back_inserter(xval));
  }

  for (unsigned int step=0; step < xval.size(); ++step) {
    double xx = xval[step];
    if (step == static_cast<unsigned int>(minPoints-1))
      xx -= 1e-9 * dx;

    yval[step]= func(&xx);
    if (_showProgress) {
      ccoutP(Plotting) << "." ;
      cout.flush() ;
    }

    if (RooAbsReal::numEvalErrors()>0) {
      if (numee>=0) {
        coutW(Plotting) << "At observable [x]=" << xx <<  " " ;
        RooAbsReal::printEvalErrors(ccoutW(Plotting),numee) ;
      }
      if (doEEVal) {
        yval[step]=eeVal ;
      }
    }
    RooAbsReal::clearEvalErrorLog() ;
  }

  const double ymax = *std::max_element(yval.begin(), yval.end());
  const double ymin = *std::min_element(yval.begin(), yval.end());
  double yrangeEst=(ymax-ymin) ;

  // store points of the coarse scan and calculate any refinements necessary
  double minDx= resolution*(xhi-xlo);
  double x1;
  double x2 = xlo;

  if (wmode==Extended) {
    // Add two points to make curve jump from 0 to yval at the left end of the plotting range.
    // This ensures that filled polygons are drawn properly. The first point needs to be to the
    // left of the second, so it's shifted by 1/1000 more than the second.
    addPoint(xlo-dx*1.001, 0);
    addPoint(xlo-dx,yval[0]) ;
  } else if (wmode==Straight) {
    addPoint(xlo-dx*0.001,0) ;
  }

  addPoint(xlo,yval[0]);

  auto iter2 = xval.begin() ;
  x1 = *iter2 ;
  int step=1 ;
  while(true) {
    x1= x2;
    ++iter2 ;
    if (iter2==xval.end()) {
      break ;
    }
    x2= *iter2 ;
    if (prec<0) {
      // If precision is <0, no attempt at recursive interpolation is made
      addPoint(x2,yval[step]) ;
    } else {
      addRange(func,x1,x2,yval[step-1],yval[step],prec*yrangeEst,minDx,numee,doEEVal,eeVal,epsilon);
    }
    step++ ;
  }
  addPoint(xhi,yval[minPoints-1]) ;

  if (wmode==Extended) {
    // Add two points to close polygon. The order matters. Since they are sorted in x later, the second
    // point is shifted by 1/1000 more than the second-to-last point.
    addPoint(xhi+dx,yval[minPoints-1]) ;
    addPoint(xhi+dx*1.001, 0);
  } else if (wmode==Straight) {
    addPoint(xhi+dx*0.001,0) ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Fill the range (x1,x2) with points calculated using func(&x). No point will
/// be added at x1, and a point will always be added at x2. The density of points
/// will be calculated so that the maximum deviation from a straight line
/// approximation is prec*(ymax-ymin) down to the specified minimum horizontal spacing.

void RooCurve::addRange(const RooAbsFunc& func, double x1, double x2,
         double y1, double y2, double minDy, double minDx,
         int numee, bool doEEVal, double eeVal, double epsilon)
{
  // Explicitly skip empty ranges to eliminate point duplication
  if (std::abs(x2-x1) <= epsilon) {
    return ;
  }

  // calculate our value at the midpoint of this range
  double xmid= 0.5*(x1+x2);
  double ymid= func(&xmid);
  if (_showProgress) {
    ccoutP(Plotting) << "." ;
    cout.flush() ;
  }

  if (RooAbsReal::numEvalErrors()>0) {
    if (numee>=0) {
      coutW(Plotting) << "At observable [x]=" << xmid <<  " " ;
      RooAbsReal::printEvalErrors(ccoutW(Plotting),numee) ;
    }
    if (doEEVal) {
      ymid=eeVal ;
    }
  }
  RooAbsReal::clearEvalErrorLog() ;

  // test if the midpoint is sufficiently close to a straight line across this interval
  double dy= ymid - 0.5*(y1+y2);
  if((xmid - x1 >= minDx) && std::abs(dy)>0 && std::abs(dy) >= minDy) {
    // fill in each subrange
    addRange(func,x1,xmid,y1,ymid,minDy,minDx,numee,doEEVal,eeVal,epsilon);
    addRange(func,xmid,x2,ymid,y2,minDy,minDx,numee,doEEVal,eeVal,epsilon);
  }
  else {
    // add the endpoint
    addPoint(x2,y2);
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Add a point with the specified coordinates. Update our y-axis limits.

void RooCurve::addPoint(double x, double y)
{
//   cout << "RooCurve("<< GetName() << ") adding point at (" << x << "," << y << ")" << endl ;
  Int_t next= GetN();
  SetPoint(next, x, y);
  updateYAxisLimits(y) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return the number of events associated with the plotable object,
/// it is always 1 for curves

double RooCurve::getFitRangeNEvt() const {
  return 1;
}


////////////////////////////////////////////////////////////////////////////////
/// Return the number of events associated with the plotable object,
/// in the given range. It is always 1 for curves

double RooCurve::getFitRangeNEvt(double, double) const
{
  return 1 ;
}


////////////////////////////////////////////////////////////////////////////////
/// Get the bin width associated with this plotable object.
/// It is alwats zero for curves

double RooCurve::getFitRangeBinW() const {
  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////

void RooCurve::printName(ostream& os) const
//
{
  // Print the name of this curve
  os << GetName() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print the title of this curve

void RooCurve::printTitle(ostream& os) const
{
  os << GetTitle() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print the class name of this curve

void RooCurve::printClassName(ostream& os) const
{
  os << ClassName() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Print the details of this curve

void RooCurve::printMultiline(ostream& os, Int_t /*contents*/, bool /*verbose*/, TString indent) const
{
  os << indent << "--- RooCurve ---" << endl ;
  Int_t n= GetN();
  os << indent << "  Contains " << n << " points" << endl;
  os << indent << "  Graph points:" << endl;
  for(Int_t i= 0; i < n; i++) {
    os << indent << std::setw(3) << i << ") x = " << fX[i] << " , y = " << fY[i] << endl;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate the chi^2/NDOF of this curve with respect to the histogram
/// 'hist' accounting nFitParam floating parameters in case the curve
/// was the result of a fit

double RooCurve::chiSquare(const RooHist& hist, Int_t nFitParam) const
{
  Int_t np = hist.GetN();

  // Find starting and ending bin of histogram based on range of RooCurve
  double xstart = GetPointX(0);
  double xstop = GetPointX(GetN()-1);

  Int_t nbin(0) ;

  ROOT::Math::KahanSum<double> chisq;
  for (int i=0 ; i<np ; i++) {

    // Retrieve histogram contents
    Point point = getPoint(hist, i);

    // Check if point is in range of curve
    if (point.x<xstart || point.x>xstop) continue ;

    double eyl = hist.GetEYlow()[i] ;
    double eyh = hist.GetEYhigh()[i] ;
    double exl = hist.GetEXlow()[i] ;
    double exh = hist.GetEXhigh()[i] ;

    // Integrate function over this bin
    double avg = average(point.x-exl,point.x+exh) ;

    // Add pull^2 to chisq
    if (point.y!=0) {
      double pull = (point.y>avg) ? ((point.y-avg)/eyl) : ((point.y-avg)/eyh) ;
      chisq += pull*pull ;
      nbin++ ;
    }
  }

  // Return chisq/nDOF
  return chisq.Sum() / (nbin-nFitParam) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return average curve value in [xFirst,xLast] by integrating curve between points
/// and dividing by xLast-xFirst

double RooCurve::average(double xFirst, double xLast) const
{
  if (xFirst>=xLast) {
    coutE(InputArguments) << "RooCurve::average(" << GetName()
           << ") invalid range (" << xFirst << "," << xLast << ")" << endl ;
    return 0 ;
  }

  // Find Y values and begin and end points
  double yFirst = interpolate(xFirst,1e-10) ;
  double yLast = interpolate(xLast,1e-10) ;

  // Find first and last mid points
  Int_t ifirst = findPoint(xFirst, std::numeric_limits<double>::infinity());
  Int_t ilast  = findPoint(xLast, std::numeric_limits<double>::infinity());

  // Make sure the midpoints are actually in the interval
  while (GetPointX(ifirst) < xFirst) {
    ++ifirst;
  }
  while (GetPointX(ilast) > xLast) {
    --ilast;
  }

  // Handle trivial scenario -- no midway points, point only at or outside given range
  if (ilast < ifirst) {
    return 0.5*(yFirst+yLast) ;
  }

  Point firstPt = getPoint(*this, ifirst);
  Point lastPt = getPoint(*this, ilast);

  // Trapezoid integration from lower edge to first midpoint
  double sum = 0.5 * (firstPt.x-xFirst)*(yFirst+firstPt.y);

  // Trapezoid integration between midpoints
  for (int i=ifirst ; i<ilast ; i++) {
    Point p1 = getPoint(*this, i) ;
    Point p2 = getPoint(*this, i+1) ;
    sum += 0.5 * (p2.x-p1.x)*(p1.y+p2.y);
  }

  // Trapezoid integration from last midpoint to upper edge
  sum += 0.5 * (xLast-lastPt.x)*(lastPt.y+yLast);
  return sum/(xLast-xFirst) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Find the nearest point to xvalue. Return -1 if distance
/// exceeds tolerance

Int_t RooCurve::findPoint(double xvalue, double tolerance) const
{
  double delta(std::numeric_limits<double>::max());
  Int_t n = GetN();
  Int_t ibest(-1) ;
  for (int i=0 ; i<n ; i++) {
    double x = GetPointX(i);
    if (std::abs(xvalue-x)<delta) {
      delta = std::abs(xvalue-x) ;
      ibest = i ;
    }
  }

  return (delta<tolerance)?ibest:-1 ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return linearly interpolated value of curve at xvalue. If distance
/// to nearest point is less than tolerance, return nearest point value
/// instead

double RooCurve::interpolate(double xvalue, double tolerance) const
{
  // Find best point
  int n = GetN() ;
  int ibest = findPoint(xvalue,1e10) ;

  // Get position of best point
  Point pbest = getPoint(*this, ibest);

  // Handle trivial case of being dead on
  if (std::abs(pbest.x-xvalue)<tolerance) {
    return pbest.y;
  }

  // Get nearest point on other side w.r.t. xvalue
  double retVal(0);
  if (pbest.x<xvalue) {
    if (ibest==n-1) {
      // Value beyond end requested -- return value of last point
      return pbest.y ;
    }
    Point pother = getPoint(*this, ibest+1);
    if (pother.x==pbest.x) return pbest.y ;
    retVal = pbest.y + (pother.y-pbest.y)*(xvalue-pbest.x)/(pother.x-pbest.x) ;

  } else {
    if (ibest==0) {
      // Value before 1st point requested -- return value of 1st point
      return pbest.y ;
    }
    Point pother = getPoint(*this, ibest-1);
    if (pother.x==pbest.x) return pbest.y ;
    retVal = pother.y + (pbest.y-pother.y)*(xvalue-pother.x)/(pbest.x-pother.x) ;
  }

  return retVal ;
}




////////////////////////////////////////////////////////////////////////////////
/// Construct filled RooCurve represented error band that captures alpha% of the variations
/// of the curves passed through argument variations, where the percentage alpha corresponds to
/// the central interval fraction of a significance Z

RooCurve* RooCurve::makeErrorBand(const vector<RooCurve*>& variations, double Z) const
{
  RooCurve* band = new RooCurve ;
  band->SetName((std::string(GetName()) + "_errorband").c_str());
  band->SetLineWidth(1) ;
  band->SetFillColor(kCyan) ;
  band->SetLineColor(kCyan) ;

  vector<double> bandLo(GetN()) ;
  vector<double> bandHi(GetN()) ;
  for (int i=0 ; i<GetN() ; i++) {
    calcBandInterval(variations,i,Z,bandLo[i],bandHi[i],false) ;
  }

  for (int i=0 ; i<GetN() ; i++) {
    band->addPoint(GetX()[i],bandLo[i]) ;
  }
  for (int i=GetN()-1 ; i>=0 ; i--) {
    band->addPoint(GetX()[i],bandHi[i]) ;
  }
  // if the axis of the old graph is alphanumeric, copy the labels to the new one as well
  if(this->GetXaxis() && this->GetXaxis()->IsAlphanumeric()){
    band->GetXaxis()->Set(this->GetXaxis()->GetNbins(),this->GetXaxis()->GetXmin(),this->GetXaxis()->GetXmax());
    for(int i=0; i<this->GetXaxis()->GetNbins(); ++i){
      band->GetXaxis()->SetBinLabel(i+1,this->GetXaxis()->GetBinLabel(i+1));
    }
  }

  return band ;
}




////////////////////////////////////////////////////////////////////////////////
/// Construct filled RooCurve represented error band represent the error added in quadrature defined by the curves arguments
/// plusVar and minusVar corresponding to one-sigma variations of each parameter. The resulting error band, combined used the correlation matrix C
/// is multiplied with the significance parameter Z to construct the equivalent of a Z sigma error band (in Gaussian approximation)

RooCurve* RooCurve::makeErrorBand(const vector<RooCurve*>& plusVar, const vector<RooCurve*>& minusVar, const TMatrixD& C, double Z) const
{

  RooCurve* band = new RooCurve ;
  band->SetName((std::string(GetName()) + "_errorband").c_str());
  band->SetLineWidth(1) ;
  band->SetFillColor(kCyan) ;
  band->SetLineColor(kCyan) ;

  vector<double> bandLo(GetN()) ;
  vector<double> bandHi(GetN()) ;
  for (int i=0 ; i<GetN() ; i++) {
    calcBandInterval(plusVar,minusVar,i,C,Z,bandLo[i],bandHi[i]) ;
  }

  for (int i=0 ; i<GetN() ; i++) {
    band->addPoint(GetX()[i],bandLo[i]) ;
  }
  for (int i=GetN()-1 ; i>=0 ; i--) {
    band->addPoint(GetX()[i],bandHi[i]) ;
  }

  // if the axis of the old graph is alphanumeric, copy the labels to the new one as well
  if(this->GetXaxis() && this->GetXaxis()->IsAlphanumeric()){
    band->GetXaxis()->Set(this->GetXaxis()->GetNbins(),this->GetXaxis()->GetXmin(),this->GetXaxis()->GetXmax());
    for(int i=0; i<this->GetXaxis()->GetNbins(); ++i){
      band->GetXaxis()->SetBinLabel(i+1,this->GetXaxis()->GetBinLabel(i+1));
    }
  }

  return band ;
}





////////////////////////////////////////////////////////////////////////////////
/// Retrieve variation points from curves

void RooCurve::calcBandInterval(const vector<RooCurve*>& plusVar, const vector<RooCurve*>& minusVar,Int_t i, const TMatrixD& C, double /*Z*/, double& lo, double& hi) const
{
  vector<double> y_plus(plusVar.size());
  vector<double> y_minus(minusVar.size());
  Int_t j(0) ;
  for (vector<RooCurve*>::const_iterator iter=plusVar.begin() ; iter!=plusVar.end() ; ++iter) {
    y_plus[j++] = (*iter)->interpolate(GetX()[i]) ;
  }
  j=0 ;
  for (vector<RooCurve*>::const_iterator iter=minusVar.begin() ; iter!=minusVar.end() ; ++iter) {
    y_minus[j++] = (*iter)->interpolate(GetX()[i]) ;
  }
  double y_cen = GetY()[i] ;
  Int_t n = j ;

  // Make vector of variations
  TVectorD F(plusVar.size()) ;
  for (j=0 ; j<n ; j++) {
    F[j] = (y_plus[j]-y_minus[j])/2 ;
  }

  // Calculate error in linear approximation from variations and correlation coefficient
  double sum = F*(C*F) ;

  lo= y_cen + sqrt(sum) ;
  hi= y_cen - sqrt(sum) ;
}



////////////////////////////////////////////////////////////////////////////////

void RooCurve::calcBandInterval(const vector<RooCurve*>& variations,Int_t i,double Z, double& lo, double& hi, bool approxGauss) const
{
  vector<double> y(variations.size()) ;
  Int_t j(0) ;
  for (vector<RooCurve*>::const_iterator iter=variations.begin() ; iter!=variations.end() ; ++iter) {
    y[j++] = (*iter)->interpolate(GetX()[i]) ;
}

  if (!approxGauss) {
    // Construct central 68% interval from variations collected at each point
    double pvalue = TMath::Erfc(Z/sqrt(2.)) ;
    Int_t delta = Int_t( y.size()*(pvalue)/2 + 0.5) ;
    sort(y.begin(),y.end()) ;
    lo = y[delta] ;
    hi = y[y.size()-delta] ;
  } else {
    // Estimate R.M.S of variations at each point and use that as Gaussian sigma
    double sum_y(0);
    double sum_ysq(0);
    for (unsigned int k=0 ; k<y.size() ; k++) {
      sum_y   += y[k] ;
      sum_ysq += y[k]*y[k] ;
    }
    sum_y /= y.size() ;
    sum_ysq /= y.size() ;

    double rms = sqrt(sum_ysq - (sum_y*sum_y)) ;
    lo = GetY()[i] - Z*rms ;
    hi = GetY()[i] + Z*rms ;
  }
}




////////////////////////////////////////////////////////////////////////////////
/// Return true if curve is identical to other curve allowing for given
/// absolute tolerance on each point compared point.

bool RooCurve::isIdentical(const RooCurve& other, double tol, bool verbose) const
{
  // Determine X range and Y range
  Int_t n= min(GetN(),other.GetN());
  double xmin(1e30);
  double xmax(-1e30);
  double ymin(1e30);
  double ymax(-1e30);
  for(Int_t i= 0; i < n; i++) {
    if (fX[i]<xmin) xmin=fX[i] ;
    if (fX[i]>xmax) xmax=fX[i] ;
    if (fY[i]<ymin) ymin=fY[i] ;
    if (fY[i]>ymax) ymax=fY[i] ;
  }
  const double Yrange=ymax-ymin ;

  bool ret(true) ;
  for(Int_t i= 2; i < n-2; i++) {
    double yTest = interpolate(other.fX[i],1e-10) ;
    double rdy = std::abs(yTest-other.fY[i])/Yrange ;
    if (rdy>tol) {
      ret = false;
      if(!verbose) continue;
      cout << "RooCurve::isIdentical[" << std::setw(3) << i << "] Y tolerance exceeded (" << std::setprecision(5) << std::setw(10) << rdy << ">" << tol << "),";
      cout << "  x,y=(" << std::right << std::setw(10) << fX[i] << "," << std::setw(10) << fY[i] << ")\tref: y="
          << std::setw(10) << other.interpolate(fX[i], 1.E-15) << ". [Nearest point from ref: ";
      auto j = other.findPoint(fX[i], 1.E10);
      std::cout << "j=" << j << "\tx,y=(" << std::setw(10) << other.fX[j] << "," << std::setw(10) << other.fY[j] << ") ]" << "\trange=" << Yrange << std::endl;
    }
  }

  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Returns sampling hints for a histogram with given boundaries. This helper
/// function is meant to be used by binned RooAbsReals to produce sampling
/// hints that are working well with RooFits plotting.

std::list<double> *
RooCurve::plotSamplingHintForBinBoundaries(std::span<const double> boundaries, double xlo, double xhi)
{
   auto hint = new std::list<double>;

   // Make sure the difference between two points around a bin boundary is
   // larger than the relative epsilon for which the RooCurve considers two
   // points as the same. Otherwise, the points right of the bin boundary would
   // be skipped.
   const double delta = (xhi - xlo) * RooCurve::relativeXEpsilon();

   // Sample points right next to the plot limits
   hint->push_back(xlo + delta);
   hint->push_back(xhi - delta);

   // Sample points very close to the left and right of the bin boundaries that
   // are strictly in between the plot limits.
   for (const double x : boundaries) {
      if (x - xlo > delta && xhi - x > delta) {
         hint->push_back(x - delta);
         hint->push_back(x + delta);
      }
   }

   hint->sort();

   return hint;
}
