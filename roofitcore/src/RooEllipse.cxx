/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooEllipse.cc,v 1.1 2002/02/05 01:23:07 davidk Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   29-Apr-2001 DK Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION [PLOT] --
// A RooEllipse is a two-dimensional ellipse that can be used to represent
// an error contour.

// #include "BaBar/BaBar.hh"

#include "RooFitCore/RooEllipse.hh"

#include <iostream.h>
#include <math.h>
#include <assert.h>

ClassImp(RooEllipse)

static const char rcsid[] =
"$Id: RooEllipse.cc,v 1.1 2002/02/05 01:23:07 davidk Exp $";

RooEllipse::RooEllipse() { }

RooEllipse::~RooEllipse() 
{
}

RooEllipse::RooEllipse(const char *name, Double_t x1, Double_t x2, Double_t s1, Double_t s2, Double_t rho, Int_t points) {
  // Create a 2-dimensional ellipse centered at (x1,x2) that represents the confidence
  // level contour for a measurement with errors (s1,s2) and correlation coefficient rho.
  // The resulting curve is defined as the unique ellipse that passes through these points:
  //
  //   (x1+rho*s1,x2+s2) , (x1-rho*s1,x2-s2) , (x1+s1,x2+rho*s2) , (x1-s1,x2-rho*s2)
  //
  // and is described by the implicit equation:
  //
  //   x*x      2*rho      y*y
  //  -----  -  -----  +  -----  =  1 - rho*rho
  //  s1*s1     s1*s2     s2*s2
  //
  // The input parameters s1,s2 must be > 0 and also |rho| <= 1.
  // The degenerate case |rho|=1 corresponds to a straight line and
  // is handled as a special case.

  SetName(name);
  SetTitle(name);

  if(s1 <= 0 || s2 <= 0) {
    cout << "RooEllipse::RooEllipse: bad parameter s1 or s2 < 0" << endl;
    return;
  }
  Double_t tmp= 1-rho*rho;
  if(tmp < 0) {
    cout << "RooEllipse::RooEllipse: bad parameter |rho| > 1" << endl;
    return;
  }

  if(tmp == 0) {
    // handle the degenerate case of |rho|=1
    SetPoint(0,x1-s1,x2-s2);
    SetPoint(1,x1+s1,x2+s2);
    setYAxisLimits(x2-s2,x2+s2);
  }
  else {
    Double_t r,phi,u1,u2,xx1,xx2,dphi(2*M_PI/points);
    for(Int_t index= 0; index < points; index++) {
      phi= index*dphi;
      u1= cos(phi)/s1;
      u2= sin(phi)/s2;
      r= sqrt(tmp/(u1*u1 - 2*rho*u1*u2 + u2*u2));
      xx1= x1 + r*u1*s1;
      xx2= x2 + r*u2*s2;
      SetPoint(index, xx1, xx2);
      if(index == 0) {
	setYAxisLimits(xx2,xx2);
	// add an extra segment to close the curve
	SetPoint(points, xx1, xx2);
      }
      else {
	updateYAxisLimits(xx2);
      }
    }
  }
}

void RooEllipse::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this histogram to the specified output stream.
  //
  //   Standard: number of entries
  //    Verbose: print points on curve

  oneLinePrint(os,*this);
  RooPlotable::printToStream(os,opt,indent);
}
