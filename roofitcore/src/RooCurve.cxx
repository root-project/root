/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooCurve.cc,v 1.1 2001/05/02 18:08:59 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   29-Apr-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// A RooCurve is a graphical representation of a real-valued function.

#include "BaBar/BaBar.hh"

#include "RooFitCore/RooCurve.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooRealFunc1D.hh"
#include "RooFitCore/RooAbsFunc1D.hh"

#include <iostream.h>
#include <iomanip.h>

ClassImp(RooCurve)

static const char rcsid[] =
"$Id: RooCurve.cc,v 1.1 2001/05/02 18:08:59 david Exp $";

RooCurve::RooCurve() {
}

RooCurve::RooCurve(const RooAbsReal &f, RooRealVar &x, Double_t prec) {
  // grab the function's name and title
  TString name("curve_");
  name.Append(f.GetName());
  SetName(name.Data());
  SetTitle(f.GetTitle());
  // bind the function to the specified real var
  RooRealFunc1D func= f(x);
  // calculate the points to add to our curve
  addPoints(func,x.getPlotMin(),x.getPlotMax(),x.getPlotBins(),prec);
}

RooCurve::RooCurve(const char *name, const char *title, const RooAbsFunc1D &func,
		   Double_t xlo, Double_t xhi, UInt_t minPoints, Double_t prec) {
  SetName(name);
  SetTitle(title);
  addPoints(func,xlo,xhi,minPoints,prec);
}

void RooCurve::addPoints(const RooAbsFunc1D &func, Double_t xlo, Double_t xhi,
			 Int_t minPoints, Double_t prec) {
  // check the inputs
  if(minPoints == 0 || xhi <= xlo) {
    cout << fName << "::addPoints: bad input (nothing added)" << endl;
    return;
  }
  // add the first point
  Double_t x1,x2= xlo,y= func(x2);
  addPoint(x2,y);

  // loop over a grid with the minimum allowed number of points
  Double_t dx= (xhi-xlo)/minPoints;
  for(Int_t step= 1; step <= minPoints; step++) {
    x1= x2;
    x2= xlo + step*dx;
    y= func(x2);
    addPoint(x2,y);
  }
}

RooCurve::~RooCurve() {
}

void RooCurve::addPoint(Double_t x, Double_t y) {
  Int_t next= GetN();
  //cout << "adding point " << next << " at ( " << x << " , " << y << " )" << endl;
  SetPoint(next, x, y);
  updateYAxisLimits(y);
}

Double_t RooCurve::addRange(const RooAbsFunc1D& func, Double_t x1, Double_t x2, Double_t y1) {
  return 0;
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
