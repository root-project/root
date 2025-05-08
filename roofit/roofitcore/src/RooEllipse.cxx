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
\file RooEllipse.cxx
\class RooEllipse
\ingroup Roofitcore

Two-dimensional ellipse that can be used to represent an error contour.
**/

#include "RooEllipse.h"
#include "TMath.h"
#include "RooMsgService.h"

#include "Riostream.h"
#include "TClass.h"
#include <cmath>


////////////////////////////////////////////////////////////////////////////////
/// Create a 2-dimensional ellipse centered at `(x1,x2)` that represents the confidence
/// level contour for a measurement with standard deviations`(s1,s2)` and correlation coefficient rho,
/// and semiaxes corresponding to `k` standard deviations.
/// It is assumed that the data are distributed according to a bivariate normal distribution:
///  \f[ p(x,y) = {1 \over 2 \pi s_1 s_2 \sqrt{1-\rho^2}} \exp (-((x-x_1)^2/s_1^2 + (y-x_2)^2/s_2^2 - 2 \rho x y/(s_1s_2))/2(1-\rho^2)) \f]
/// \see ROOT::Math::bigaussian_pdf
/// Or in a split form:
/// \f[ p(z(x,y)) = {1 \over 2 \pi s_1 s_2 \sqrt{1-\rho^2}} \exp (-z/2(1-\rho^2)) \f]
/// with:
///  \f[ z = ((x-x_1)^2/s_1^2 + (y-x_2)^2/s_2^2 - 2 \rho x y/(s_1s_2)) \f]
/// As demonstrated in https://root-forum.cern.ch/t/drawing-convergence-ellipse-in-2d/61936/10, the
/// "oriented" ellipse with semi-axis = (k * s_1, k * s_2) includes 39% (for k = 1) of
/// all data, this confidence ellipse can thus be interpreted as a confidence contour level.
/// This ellipse can be described by the implicit equation `z/(1-rho*rho) = k^2 = - 2 * nll ratio`, or explictly:
///
///   x*x      2*rho*x*y      y*y
///  -----  -  ---------  +  -----  =  k*k * (1 - rho*rho)
///  s1*s1       s1*s2       s2*s2
///
/// The input parameters s1,s2,k must be > 0 and also |rho| <= 1.
/// The degenerate case |rho|=1 corresponds to a straight line and
/// is handled as a special case.
/// \warning The default value of k = 1 (semiaxes at +/-sigma) corresponds to the 39% CL contour. For a 1-sigma confidence
/// level (68%), set instead k ~ 1.51 (semiaxes at +/-1.51sigma).

RooEllipse::RooEllipse(const char *name, double x1, double x2, double s1, double s2, double rho, Int_t points, double k)
{
  SetName(name);
  SetTitle(name);

  if(s1 <= 0 || s2 <= 0 || k <= 0) {
    coutE(InputArguments) << "RooEllipse::RooEllipse: bad parameter s1, s2 or k <= 0" << std::endl;
    return;
  }
  double tmp = k*k*(1-rho*rho);
  if(tmp < 0) {
    coutE(InputArguments) << "RooEllipse::RooEllipse: bad parameter |rho| > 1" << std::endl;
    return;
  }

  if(tmp == 0) {
    // handle the degenerate case of |rho|=1
    SetPoint(0,x1-s1,x2-s2);
    SetPoint(1,x1+s1,x2+s2);
    setYAxisLimits(x2-s2,x2+s2);
  }
  else {
    double r;
    double psi;
    double phi;
    double u1;
    double u2;
    double xx1;
    double xx2;
    double dphi(2 * TMath::Pi() / points);
    for(Int_t index= 0; index < points; index++) {
      phi= index*dphi;
      // adjust the angular spacing of the points for the aspect ratio
      psi= atan2(s2*sin(phi),s1*cos(phi));
      u1= cos(psi)/s1;
      u2= sin(psi)/s2;
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



////////////////////////////////////////////////////////////////////////////////
/// Print name of ellipse on ostream

void RooEllipse::printName(std::ostream& os) const
{
  os << GetName() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print title of ellipse on ostream

void RooEllipse::printTitle(std::ostream& os) const
{
  os << GetName() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print class name of ellipse on ostream

void RooEllipse::printClassName(std::ostream& os) const
{
  os << ClassName() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Print detailed multi line information on ellipse on ostreamx

void RooEllipse::printMultiline(std::ostream& os, Int_t contents, bool verbose, TString indent) const
{
  RooPlotable::printMultiline(os,contents,verbose,indent);
  for(Int_t index=0; index < fNpoints; index++) {
    os << indent << "Point [" << index << "] is at (" << fX[index] << "," << fY[index] << ")" << std::endl;
  }
}
