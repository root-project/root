/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooCurve.cxx,v 1.49 2007/05/11 09:11:58 verkerke Exp $
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
// A RooCurve is a one-dimensional graphical representation of a real-valued function.
// A curve is approximated by straight line segments with endpoints chosen to give
// a "good" approximation to the true curve. The goodness of the approximation is
// controlled by a precision and a resolution parameter. To view the points where
// a function y(x) is actually evaluated to approximate a smooth curve, use:
//
//  RooPlot *p= y.plotOn(x.frame());
//  p->getAttMarker("curve_y")->SetMarkerStyle(20);
//  p->setDrawOptions("curve_y","PL");
//  p->Draw();


#include "RooFit.h"

#include "RooCurve.h"
#include "RooCurve.h"
#include "RooHist.h"
#include "RooAbsReal.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooRealIntegral.h"
#include "RooRealBinding.h"
#include "RooScaledFunc.h"

#include "Riostream.h"
#include <iomanip>
#include <math.h>
#include <assert.h>
#include <deque>
#include <algorithm>

ClassImp(RooCurve)

RooCurve::RooCurve() {
  initialize();
}

RooCurve::RooCurve(const RooAbsReal &f, RooAbsRealLValue &x, Double_t xlo, Double_t xhi, Int_t xbins,
		   Double_t scaleFactor, const RooArgSet *normVars, Double_t prec, Double_t resolution,
		   Bool_t shiftToZero, WingMode wmode) {
  // Create a 1-dim curve of the value of the specified real-valued expression
  // as a function of x. Use the optional precision parameter to control
  // how precisely the smooth curve is rasterized. Use the optional argument set
  // to specify how the expression should be normalized. Use the optional scale
  // factor to rescale the expression after normalization.
  // If shiftToZero is set, the entire curve is shift down to make the lowest
  // point in of the curve go through zero.

  // grab the function's name and title
  TString name("curve_");
  name.Append(f.GetName());
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

  RooAbsFunc *funcPtr = 0;
  RooAbsFunc *rawPtr  = 0;
  funcPtr= f.bindVars(x,normVars,kTRUE);

  // apply a scale factor if necessary
  if(scaleFactor != 1) {
    rawPtr= funcPtr;
    funcPtr= new RooScaledFunc(*rawPtr,scaleFactor);
  }
  assert(0 != funcPtr);

  // calculate the points to add to our curve
  Double_t prevYMax = getYAxisMax() ;
  addPoints(*funcPtr,xlo,xhi,xbins+1,prec,resolution,wmode);
  initialize();

  // cleanup
  delete funcPtr;
  if(rawPtr) delete rawPtr;
  if (shiftToZero) shiftCurveToZero(prevYMax) ;

  // Adjust limits
  Int_t i ;
  for (i=0 ; i<GetN() ; i++) {    
    Double_t x,y ;
    GetPoint(i,x,y) ;
    updateYAxisLimits(y);
  }
}



RooCurve::RooCurve(const char *name, const char *title, const RooAbsFunc &func,
		   Double_t xlo, Double_t xhi, UInt_t minPoints, Double_t prec, Double_t resolution,
		   Bool_t shiftToZero, WingMode wmode) {
  SetName(name);
  SetTitle(title);
  Double_t prevYMax = getYAxisMax() ;
  addPoints(func,xlo,xhi,minPoints+1,prec,resolution,wmode);  
  initialize();
  if (shiftToZero) shiftCurveToZero(prevYMax) ;

  // Adjust limits
  Int_t i ;
  for (i=0 ; i<GetN() ; i++) {    
    Double_t x,y ;
    GetPoint(i,x,y) ;
    updateYAxisLimits(y);
  }
}


RooCurve::RooCurve(const char* name, const char* title, const RooCurve& c1, const RooCurve& c2, Double_t scale1, Double_t scale2) 
{
  initialize() ;
  SetName(name) ;
  SetTitle(title) ;

  // Make deque of points in X
  deque<Double_t> pointList ;
  Double_t x,y ;

  // Add X points of C1
  Int_t i1,n1 = c1.GetN() ;
  for (i1=0 ; i1<n1 ; i1++) {
    const_cast<RooCurve&>(c1).GetPoint(i1,x,y) ;
    pointList.push_back(x) ;
  }

  // Add X points of C2
  Int_t i2,n2 = c2.GetN() ;
  for (i2=0 ; i2<n2 ; i2++) {
    const_cast<RooCurve&>(c2).GetPoint(i2,x,y) ;
    pointList.push_back(x) ;
  }
  
  // Sort X points
  sort(pointList.begin(),pointList.end()) ;

  // Loop over X points
  deque<double>::iterator iter ;
  Double_t last(-RooNumber::infinity) ;
  for (iter=pointList.begin() ; iter!=pointList.end() ; ++iter) {

    if ((*iter-last)>1e-10) {      
      // Add OR of points to new curve, skipping duplicate points within tolerance
      addPoint(*iter,scale1*c1.interpolate(*iter)+scale2*c2.interpolate(*iter)) ;
    }
    last = *iter ;
  }
}



RooCurve::~RooCurve() 
{
}



void RooCurve::initialize() 
{
  // Perform initialization that is common to all constructors.

  // set default line width in pixels
  SetLineWidth(3);
  // set default line color
  SetLineColor(kBlue);
}


void RooCurve::shiftCurveToZero(Double_t prevYMax) 
  // Find lowest point in curve and move all points in curve so that
  // lowest point will go exactly through zero
{
  Int_t i ;
  Double_t minVal(1e30) ;
  Double_t maxVal(-1e30) ;

  // First iteration, find current lowest point
  for (i=1 ; i<GetN()-1 ; i++) {
    Double_t x,y ;
    GetPoint(i,x,y) ;
    if (y<minVal) minVal=y ;
    if (y>maxVal) maxVal=y ;
  }

  // Second iteration, lower all points by minVal
  for (i=1 ; i<GetN()-1 ; i++) {
    Double_t x,y ;
    GetPoint(i,x,y) ;
    SetPoint(i,x,y-minVal) ;
  }

  // Check if y-axis range needs readjustment
  if (getYAxisMax()>prevYMax) {
    Double_t newMax = maxVal - minVal ;
    setYAxisLimits(getYAxisMin(), newMax<prevYMax ? prevYMax : newMax) ;
  }
}



void RooCurve::addPoints(const RooAbsFunc &func, Double_t xlo, Double_t xhi,
			 Int_t minPoints, Double_t prec, Double_t resolution, WingMode wmode) {
  // Add points calculated with the specified function, over the range (xlo,xhi).
  // Add at least minPoints equally spaced points, and add sufficient points so that
  // the maximum deviation from the final straight-line segements is prec*(ymax-ymin),
  // down to a minimum horizontal spacing of resolution*(xhi-xlo).

  // check the inputs
  if(!func.isValid()) {
    cout << fName << "::addPoints: input function is not valid" << endl;
    return;
  }
  if(minPoints <= 0 || xhi <= xlo) {
    cout << fName << "::addPoints: bad input (nothing added)" << endl;
    return;
  }

  // Perform a coarse scan of the function to estimate its y range.
  // Save the results so we do not have to re-evaluate at the scan points.
  Double_t *yval= new Double_t[minPoints];
  assert(0 != yval);
  Double_t x,dx= (xhi-xlo)/(minPoints-1.);

  Int_t step;
  Double_t ymax(-1e30), ymin(1e30) ;
  for(step= 0; step < minPoints; step++) {
    x= xlo + step*dx;
    if (step==minPoints-1) x-=1e-15 ;
    yval[step]= func(&x);
    if (yval[step]>ymax) ymax=yval[step] ;
    if (yval[step]<ymin) ymin=yval[step] ;
  }
  Double_t yrangeEst=(ymax-ymin) ;

  // store points of the coarse scan and calculate any refinements necessary
  Double_t minDx= resolution*(xhi-xlo);
  Double_t x1,x2= xlo;

  if (wmode==Extended) {
    addPoint(xlo-dx,0) ;
    addPoint(xlo-dx,yval[0]) ;
  } else if (wmode==Straight) {
    addPoint(xlo,0) ;
  }

  addPoint(xlo,yval[0]);
  for(step= 1; step < minPoints; step++) {
    x1= x2;
    x2= xlo + step*dx;
    addRange(func,x1,x2,yval[step-1],yval[step],prec*yrangeEst,minDx);
  }
  addPoint(xhi,yval[minPoints-1]) ;

  if (wmode==Extended) {
    addPoint(xhi+dx,yval[minPoints-1]) ;
    addPoint(xhi+dx,0) ;
  } else if (wmode==Straight) {
    addPoint(xhi,0) ;
  }

  // cleanup
  delete [] yval;
}

void RooCurve::addRange(const RooAbsFunc& func, Double_t x1, Double_t x2,
			Double_t y1, Double_t y2, Double_t minDy, Double_t minDx) {
  // Fill the range (x1,x2) with points calculated using func(&x). No point will
  // be added at x1, and a point will always be added at x2. The density of points
  // will be calculated so that the maximum deviation from a straight line
  // approximation is prec*(ymax-ymin) down to the specified minimum horizontal spacing.

  // calculate our value at the midpoint of this range
  Double_t xmid= 0.5*(x1+x2);
  Double_t ymid= func(&xmid);
  // test if the midpoint is sufficiently close to a straight line across this interval
  Double_t dy= ymid - 0.5*(y1+y2);
  if((xmid - x1 >= minDx) && fabs(dy)>0 && fabs(dy) >= minDy) {
    // fill in each subrange
    addRange(func,x1,xmid,y1,ymid,minDy,minDx);
    addRange(func,xmid,x2,ymid,y2,minDy,minDx);
  }
  else {
    // add the endpoint
    addPoint(x2,y2);
  }
}

void RooCurve::addPoint(Double_t x, Double_t y) {
  // Add a point with the specified coordinates. Update our y-axis limits.

  // cout << "RooCurve("<< GetName() << ") adding point at (" << x << "," << y << ")" << endl ;
  Int_t next= GetN();
  SetPoint(next, x, y);
}

Double_t RooCurve::getFitRangeNEvt() const {
  return 1;
}

Double_t RooCurve::getFitRangeNEvt(Double_t, Double_t) const 
{
  return 1 ;
}

Double_t RooCurve::getFitRangeBinW() const {
  return 0 ;
}

void RooCurve::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this histogram to the specified output stream.
  //
  //   Standard: number of entries
  //    Verbose: print points on curve

  oneLinePrint(os,*this);
  RooPlotable::printToStream(os,opt,indent);
  if(opt >= Standard) {
    os << indent << "--- RooCurve ---" << endl;
    Int_t n= GetN();
    os << indent << "  Contains " << n << " points" << endl;
    if(opt >= Verbose) {
      os << indent << "  Graph points:" << endl;
      for(Int_t i= 0; i < n; i++) {
	os << indent << setw(3) << i << ") x = " << fX[i] << " , y = " << fY[i] << endl;
      }
    }
  }
}


Double_t RooCurve::chiSquare(const RooHist& hist, Int_t nFitParam) const 
{
  Int_t i,np = hist.GetN() ;
  Double_t x,y,eyl,eyh ;
  Double_t hbinw2 = hist.getNominalBinWidth()/2 ;

  // Find starting and ending bin of histogram based on range of RooCurve
  Double_t xstart,xstop ;

#if ROOT_VERSION_CODE >= ROOT_VERSION(4,0,1)
  GetPoint(0,xstart,y) ;
  GetPoint(GetN()-1,xstop,y) ;
#else
  const_cast<RooCurve*>(this)->GetPoint(0,xstart,y) ;
  const_cast<RooCurve*>(this)->GetPoint(GetN()-1,xstop,y) ;
#endif

  Int_t nbin(0) ;

  Double_t chisq(0) ;
  for (i=0 ; i<np ; i++) {   

    // Retrieve histogram contents
    ((RooHist&)hist).GetPoint(i,x,y) ;

    // Check if point is in range of curve
    if (x<xstart || x>xstop) continue ;

    nbin++ ;
    eyl = hist.GetEYlow()[i] ;
    eyh = hist.GetEYhigh()[i] ;

    // Integrate function over this bin
    Double_t avg = average(x-hbinw2,x+hbinw2) ;

    // Add pull^2 to chisq
    if (y!=0) {      
      Double_t pull = (y>avg) ? ((y-avg)/eyl) : ((y-avg)/eyh) ;
      chisq += pull*pull ;
    }
  }

  // Return chisq/nDOF 
  return chisq / (nbin-nFitParam) ;
}



Double_t RooCurve::average(Double_t xFirst, Double_t xLast) const
{
  // Average curve between given values by integrating curve between points
  // and dividing by xLast-xFirst

  if (xFirst>=xLast) {
    cout << "RooCurve::average(" << GetName() 
	 << ") invalid range (" << xFirst << "," << xLast << ")" << endl ;
    return 0 ;
  }

  // Find Y values and begin and end points
  Double_t yFirst = interpolate(xFirst,1e-10) ;
  Double_t yLast = interpolate(xLast,1e-10) ;

  // Find first and last mid points
  Int_t ifirst = findPoint(xFirst,1e10) ;
  Int_t ilast  = findPoint(xLast,1e10) ;
  Double_t xFirstPt,yFirstPt,xLastPt,yLastPt ;
  const_cast<RooCurve&>(*this).GetPoint(ifirst,xFirstPt,yFirstPt) ;
  const_cast<RooCurve&>(*this).GetPoint(ilast,xLastPt,yLastPt) ;

  Double_t tolerance=1e-3*(xLast-xFirst) ;

  // Handle trivial scenario -- no midway points, point only at or outside given range
  if (ilast-ifirst==1 &&(xFirstPt-xFirst)<-1*tolerance && (xLastPt-xLast)>tolerance) {
    return 0.5*(yFirst+yLast) ;
  }
 
  // If first point closest to xFirst is at xFirst or before xFirst take the next point
  // as the first midway point   
  if ((xFirstPt-xFirst)<-1*tolerance) {
    ifirst++ ;
    const_cast<RooCurve&>(*this).GetPoint(ifirst,xFirstPt,yFirstPt) ;
  }
  
  // If last point closest to yLast is at yLast or beyond yLast the the previous point
  // as the last midway point
  if ((xLastPt-xLast)>tolerance) {
    ilast-- ;
    const_cast<RooCurve&>(*this).GetPoint(ilast,xLastPt,yLastPt) ;
  }

  Double_t sum(0),x1,y1,x2,y2 ;

  // Trapezoid integration from lower edge to first midpoint
  sum += (xFirstPt-xFirst)*(yFirst+yFirstPt)/2 ;

  // Trapezoid integration between midpoints
  Int_t i ;
  for (i=ifirst ; i<ilast ; i++) {
    const_cast<RooCurve&>(*this).GetPoint(i,x1,y1) ;
    const_cast<RooCurve&>(*this).GetPoint(i+1,x2,y2) ;
    sum += (x2-x1)*(y1+y2)/2 ;
  }

  // Trapezoid integration from last midpoint to upper edge 
  sum += (xLast-xLastPt)*(yLastPt+yLast)/2 ;
  return sum/(xLast-xFirst) ;
}



Int_t RooCurve::findPoint(Double_t xvalue, Double_t tolerance) const
{
  Double_t delta(999.),x,y ;
  Int_t i,n = GetN() ;
  Int_t ibest(-1) ;
  for (i=0 ; i<n ; i++) {
    ((RooCurve&)*this).GetPoint(i,x,y) ;
    if (fabs(xvalue-x)<delta) {
      delta = fabs(xvalue-x) ;
      ibest = i ;
    }
  }

  return (delta<tolerance)?ibest:-1 ;
}

Double_t RooCurve::interpolate(Double_t xvalue, Double_t tolerance) const
{
  // Find best point
  int n = GetN() ;
  int ibest = findPoint(xvalue,1e10) ;
  
  // Get position of best point
  Double_t xbest, ybest ;
  const_cast<RooCurve*>(this)->GetPoint(ibest,xbest,ybest) ;

  // Handle trivial case of being dead on
  if (fabs(xbest-xvalue)<tolerance) {
    return ybest ;
  }

  // Get nearest point on other side w.r.t. xvalue
  Double_t xother,yother, retVal(0) ;
  if (xbest<xvalue) {
    if (ibest==n-1) {
      // Value beyond end requested -- return value of last point
      return ybest ;
    }
    const_cast<RooCurve*>(this)->GetPoint(ibest+1,xother,yother) ;        
    if (xother==xbest) return ybest ;
    retVal = ybest + (yother-ybest)*(xvalue-xbest)/(xother-xbest) ; 

  } else {
    if (ibest==0) {
      // Value before 1st point requested -- return value of 1st point
      return ybest ;
    }
    const_cast<RooCurve*>(this)->GetPoint(ibest-1,xother,yother) ;    
    if (xother==xbest) return ybest ;
    retVal = yother + (ybest-yother)*(xvalue-xother)/(xbest-xother) ;
  }
 
  return retVal ;
}
