/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   RW, Ruddick William  UC Colorado        wor@slac.stanford.edu           *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/** \class RooBukinPdf
    \ingroup Roofit

The RooBukinPdf implements the NovosibirskA function. For the parameters, see
RooBukinPdf().

Credits:
May 26, 2003.
A. Bukin, Budker INP, Novosibirsk

\image html RooBukin.png
**/

#include "RooFit.h"

#include <math.h>


#include "RooBukinPdf.h"
#include "RooRealVar.h"
#include "TMath.h"

using namespace std;

ClassImp(RooBukinPdf);

////////////////////////////////////////////////////////////////////////////////
/// Construct a Bukin PDF.
/// \param name  The name of the PDF for RooFit's bookeeping.
/// \param title The title for e.g. plotting it.
/// \param _x    The variable.
/// \param _Xp   The peak position.
/// \param _sigp The peak width as FWHM divided by 2*sqrt(2*log(2))=2.35
/// \param _xi   Peak asymmetry. Use values around 0.
/// \param _rho1 Left tail. Use slightly negative starting values.
/// \param _rho2 Right tail. Use slightly positive starting values.
RooBukinPdf::RooBukinPdf(const char *name, const char *title,
          RooAbsReal& _x,    RooAbsReal& _Xp,
          RooAbsReal& _sigp, RooAbsReal& _xi,
          RooAbsReal& _rho1, RooAbsReal& _rho2) :
  // The two addresses refer to our first dependent variable and
  // parameter, respectively, as declared in the rdl file
  RooAbsPdf(name, title),
  x("x","x",this,_x),
  Xp("Xp","Xp",this,_Xp),
  sigp("sigp","sigp",this,_sigp),
  xi("xi","xi",this,_xi),
  rho1("rho1","rho1",this,_rho1),
  rho2("rho2","rho2",this,_rho2)
{

}

////////////////////////////////////////////////////////////////////////////////
/// Copy a Bukin PDF.
RooBukinPdf::RooBukinPdf(const RooBukinPdf& other, const char *name):
  RooAbsPdf(other,name),
  x("x",this,other.x),
  Xp("Xp",this,other.Xp),
  sigp("sigp",this,other.sigp),
  xi("xi",this,other.xi),
  rho1("rho1",this,other.rho1),
  rho2("rho2",this,other.rho2)

{

}

////////////////////////////////////////////////////////////////////////////////
/// Implementation

Double_t RooBukinPdf::evaluate() const
{
  const double r1 = xi/sqrt(xi*xi+1);
        double r2 = 0;
  const double r3 = log(2.);
  const double r4 = sqrt(xi*xi+1);
  const double r5 = fabs(xi) > exp(-6.) ? xi/log(r4+xi) : 1.;

  const double hp = sigp * 2*sqrt(2*log(2.));

  const double x1 = Xp + (hp / 2) * (r1-1);
  const double x2 = Xp + (hp / 2) * (r1+1);


  if(x < x1){
    //--- Left Side
    r2 = rho1 * (x-x1)/(Xp-x1)*(x-x1)/(Xp-x1) - r3 + 4 * r3 * (x-x1)/hp * r5 * r4/(r4-xi)/(r4-xi);
  } else if(x >= x2) {
    //--- Right Side
    r2 = rho2 * (x-x2)/(Xp-x2)*(x-x2)/(Xp-x2) - r3 - 4 * r3 * (x-x2)/hp * r5 * r4/(r4+xi)/(r4+xi);
  } else {
    //--- Center
    if(TMath::Abs(xi) > exp(-6.)) {
      r2=log(1 + 4 * xi * r4 * (x-Xp)/hp)/log(1+2*xi*(xi-r4));
      r2=-r3*r2*r2;
    }
    else{
      r2=-4*r3* ((x-Xp)/hp)*((x-Xp)/hp);
    }
  }

  const double fit_result = fabs(r2) > 100 ? 0. : exp(r2);

  return fit_result;
}
