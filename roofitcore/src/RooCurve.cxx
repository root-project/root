/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCurve.cc,v 1.8 2001/08/03 21:44:57 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   29-Apr-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// A RooCurve is a graphical representation of a real-valued function.

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
"$Id: RooCurve.cc,v 1.8 2001/08/03 21:44:57 david Exp $";

RooCurve::RooCurve() {
  initialize();
}

RooCurve::RooCurve(const RooAbsReal &f, RooRealVar &x, Double_t scaleFactor,
		   const RooArgSet *normVars, Double_t prec) {
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
  RooRealIntegral *projected(0);
  if(0 != normVars) {
    // calculate our normalization factor over x, if requested
    RooArgSet vars(*normVars);
    RooAbsArg *found= vars.find(x.GetName());
    if(found) {
      // calculate our normalization factor over all vars including x
      RooRealIntegral normFunc("normFunc","normFunc",f,vars);
      if(!normFunc.isValid()) {
	cout << "RooPlot: cannot normalize ";
	f.Print("1");
	return;
      }
      scaleFactor/= normFunc.getVal();
      // remove x from the set of vars to be projected
      vars.remove(*found);
    }
    // project out any remaining normalization variables
    if(vars.GetSize() > 0) {
      projected= new RooRealIntegral(TString(f.GetName()).Append("Projected"),
				     TString(f.GetTitle()).Append(" (Projected)"),
				     f,vars);
      if(!projected->isValid()) return; // can this happen if normFunc isn't valid??
      funcPtr= projected->bindVars(x);
    }
  }
  if(0 == funcPtr) funcPtr= f.bindVars(x);
  // apply a scale factor if necessary
  if(scaleFactor != 1) {
    rawPtr= funcPtr;
    funcPtr= new RooScaledFunc(*rawPtr,scaleFactor);
  }
  assert(0 != funcPtr);

  // calculate the points to add to our curve
  addPoints(*funcPtr,x.getPlotMin(),x.getPlotMax(),x.getPlotBins(),prec);
  initialize();

  // cleanup
  delete funcPtr;
  if(rawPtr) delete rawPtr;
  if(projected) delete projected;
}

RooCurve::RooCurve(const char *name, const char *title, const RooAbsFunc &func,
		   Double_t xlo, Double_t xhi, UInt_t minPoints, Double_t prec) {
  SetName(name);
  SetTitle(title);
  addPoints(func,xlo,xhi,minPoints,prec);
  initialize();
}

void RooCurve::initialize() {
  // Perform initialization that is common to all constructors.

  // set default line width in pixels
  SetLineWidth(3);
  // set default line color
  SetLineColor(kBlue);
}

void RooCurve::addPoints(const RooAbsFunc &func, Double_t xlo, Double_t xhi,
			 Int_t minPoints, Double_t prec) {
  // Add points calculated with the specified function, over the range (xlo,xhi).
  // Add at least minPoints equally spaced points, and add sufficient points so that
  // the maximum deviation from the final straight-line segements is prec*(ymax-ymin).

  // check the inputs
  if(!func.isValid()) {
    cout << fName << "::addPoints: input function is not valid" << endl;
    return;
  }
  if(minPoints == 0 || xhi <= xlo) {
    cout << fName << "::addPoints: bad input (nothing added)" << endl;
    return;
  }
  // add the first point
  Double_t x1,y1,x2= xlo,y2= func(&x2);
  addPoint(x2,y2);

  // loop over a grid with the minimum allowed number of points
  Double_t dx= (xhi-xlo)/minPoints;
  for(Int_t step= 1; step <= minPoints; step++) {
    x1= x2;
    y1= y2;
    x2= xlo + step*dx;
    y2= func(&x2);    
    addRange(func,x1,x2,y1,y2,prec);
  }
}

RooCurve::~RooCurve() {
}

Double_t RooCurve::getFitRangeNorm() const {
  return 1;
}

void RooCurve::addPoint(Double_t x, Double_t y) {
  // Add a point with the specified coordinates. Update our y-axis limits.

  Int_t next= GetN();
  SetPoint(next, x, y);
  updateYAxisLimits(y);
}

void RooCurve::addRange(const RooAbsFunc& func, Double_t x1, Double_t x2,
			Double_t y1, Double_t y2, Double_t prec) {
  // Fill the range (x1,x2) with points calculated using func(&x). No point will
  // be added at x1, and a point will always be added at x2. The density of points
  // will be calculated so that the maximum deviation from a straight line
  // approximation is prec*(ymax-ymin).

  // calculate our value at the midpoint of this range
  Double_t xmid= 0.5*(x1+x2);
  Double_t ymid= func(&xmid);
  // test if the midpoint is sufficiently close to a straight line across this interval
  Double_t dy= ymid - 0.5*(y1+y2);
  if(fabs(dy) >= prec*(getYAxisMax()-getYAxisMin())) {
    // fill in each subrange
    addRange(func,x1,xmid,y1,ymid,prec);
    addRange(func,xmid,x2,ymid,y2,prec);
  }
  else {
    // add the endpoint
    addPoint(x2,y2);
  }
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
