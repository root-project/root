/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCurve.cc,v 1.22 2001/10/03 16:16:31 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   29-Apr-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
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

// #include "BaBar/BaBar.hh"

#include "RooFitCore/RooCurve.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooRealIntegral.hh"
#include "RooFitCore/RooRealBinding.hh"
#include "RooFitCore/RooScaledFunc.hh"

#include <iostream.h>
#include <iomanip.h>
#include <math.h>
#include <assert.h>

ClassImp(RooCurve)

static const char rcsid[] =
"$Id: RooCurve.cc,v 1.22 2001/10/03 16:16:31 verkerke Exp $";

RooCurve::RooCurve() {
  initialize();
}

RooCurve::RooCurve(const RooAbsReal &f, RooAbsRealLValue &x, Double_t scaleFactor,
		   const RooArgSet *normVars, Double_t prec, Double_t resolution) {
  // Create a 1-dim curve of the value of the specified real-valued expression
  // as a function of x. Use the optional precision parameter to control
  // how precisely the smooth curve is rasterized. Use the optional argument set
  // to specify how the expression should be normalized. Use the optional scale
  // factor to rescale the expression after normalization.

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

  RooAbsFunc *funcPtr(0),*rawPtr(0);
  funcPtr= f.bindVars(x,normVars);

  // apply a scale factor if necessary
  if(scaleFactor != 1) {
    rawPtr= funcPtr;
    funcPtr= new RooScaledFunc(*rawPtr,scaleFactor);
  }
  assert(0 != funcPtr);

  // calculate the points to add to our curve
  addPoints(*funcPtr,x.getPlotMin(),x.getPlotMax(),x.getPlotBins()+1,prec,resolution);
  initialize();

  // cleanup
  delete funcPtr;
  if(rawPtr) delete rawPtr;
}



RooCurve::RooCurve(const char *name, const char *title, const RooAbsFunc &func,
		   Double_t xlo, Double_t xhi, UInt_t minPoints, Double_t prec, Double_t resolution) {
  SetName(name);
  SetTitle(title);
  addPoints(func,xlo,xhi,minPoints,prec,resolution);
  initialize();
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



void RooCurve::addPoints(const RooAbsFunc &func, Double_t xlo, Double_t xhi,
			 Int_t minPoints, Double_t prec, Double_t resolution) {
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
  for(Int_t step= 0; step < minPoints; step++) {
    x= xlo + step*dx;
    yval[step]= func(&x);
    updateYAxisLimits(yval[step]);
  }

  // store points of the coarse scan and calculate any refinements necessary
  Double_t minDx= resolution*(xhi-xlo);
  Double_t x1,x2= xlo;
  addPoint(xlo,yval[0]);
  for(Int_t step= 1; step < minPoints; step++) {
    x1= x2;
    x2= xlo + step*dx;
    addRange(func,x1,x2,yval[step-1],yval[step],prec,minDx);
  }

  // cleanup
  delete [] yval;
}

void RooCurve::addRange(const RooAbsFunc& func, Double_t x1, Double_t x2,
			Double_t y1, Double_t y2, Double_t prec, Double_t minDx) {
  // Fill the range (x1,x2) with points calculated using func(&x). No point will
  // be added at x1, and a point will always be added at x2. The density of points
  // will be calculated so that the maximum deviation from a straight line
  // approximation is prec*(ymax-ymin) down to the specified minimum horizontal spacing.

  // calculate our value at the midpoint of this range
  Double_t xmid= 0.5*(x1+x2);
  Double_t ymid= func(&xmid);
  // test if the midpoint is sufficiently close to a straight line across this interval
  Double_t dy= ymid - 0.5*(y1+y2);
  if((xmid - x1 >= minDx) && fabs(dy)>0 && fabs(dy) >= prec*(getYAxisMax()-getYAxisMin())) {
    // fill in each subrange
    updateYAxisLimits(ymid);
    addRange(func,x1,xmid,y1,ymid,prec,minDx);
    addRange(func,xmid,x2,ymid,y2,prec,minDx);
  }
  else {
    // add the endpoint
    addPoint(x2,y2);
  }
}

void RooCurve::addPoint(Double_t x, Double_t y) {
  // Add a point with the specified coordinates. Update our y-axis limits.

  Int_t next= GetN();
  SetPoint(next, x, y);
}

Double_t RooCurve::getFitRangeNEvt() const {
  return 1;
}

Double_t RooCurve::getFitRangeBinW() const {
  return 1;
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
