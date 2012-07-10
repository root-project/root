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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// RooBukinPdf implements the NovosibirskA function 
// END_HTML
//

// Original Fortran Header below
/*****************************************************************************
 * Fitting function for asymmetric peaks with 6 free parameters:	     *
 *     Ap   - peak value						     *
 *     Xp   - peak position						     *
 *     sigp - FWHM divided by 2*sqrt(2*log(2))=2.35			     *
 *     xi   - peak asymmetry parameter					     *
 *     rho1 - parameter of the "left tail"				     *
 *     rho2 - parameter of the "right tail"				     *
 *   ---------------------------------------------			     *
 *       May 26, 2003							     *
 *       A.Bukin, Budker INP, Novosibirsk				     *
 *       Documentation:							     *
 *       http://www.slac.stanford.edu/BFROOT/www/Organization/CollabMtgs/2003/detJuly2003/Tues3a/bukin.ps 
 *   -------------------------------------------			     *
 *****************************************************************************/

#include "RooFit.h"

#include <math.h>


#include "RooBukinPdf.h"
#include "RooRealVar.h"
#include "TMath.h"

using namespace std;

ClassImp(RooBukinPdf)



//_____________________________________________________________________________
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
  // Constructor
  consts = 2*sqrt(2*log(2.));
}



//_____________________________________________________________________________
RooBukinPdf::RooBukinPdf(const RooBukinPdf& other, const char *name):
  RooAbsPdf(other,name),
  x("x",this,other.x),
  Xp("Xp",this,other.Xp),
  sigp("sigp",this,other.sigp),
  xi("xi",this,other.xi),
  rho1("rho1",this,other.rho1),
  rho2("rho2",this,other.rho2)

{
  // Copy constructor
  consts = 2*sqrt(2*log(2.));
}



//_____________________________________________________________________________
Double_t RooBukinPdf::evaluate() const 
{
  // Implementation 

  double r1=0,r2=0,r3=0,r4=0,r5=0,hp=0;
  double x1 = 0,x2 = 0;
  double fit_result = 0;
  
  hp=sigp*consts;
  r3=log(2.);
  r4=sqrt(TMath::Power(xi,2)+1);
  r1=xi/r4;  

  if(TMath::Abs(xi) > exp(-6.)){
    r5=xi/log(r4+xi);
  }
  else
    r5=1;
    
  x1 = Xp + (hp / 2) * (r1-1);
  x2 = Xp + (hp / 2) * (r1+1);
  
  //--- Left Side
  if(x < x1){
    r2=rho1*TMath::Power((x-x1)/(Xp-x1),2)-r3 + 4 * r3 * (x-x1)/hp * r5 * r4/TMath::Power((r4-xi),2);
  }


  //--- Center
  else if(x < x2) {
    if(TMath::Abs(xi) > exp(-6.)) {
      r2=log(1 + 4 * xi * r4 * (x-Xp)/hp)/log(1+2*xi*(xi-r4));
      r2=-r3*(TMath::Power(r2,2));
    }
    else{
      r2=-4*r3*TMath::Power(((x-Xp)/hp),2);  
    }
  }
  

  //--- Right Side
  else {
    r2=rho2*TMath::Power((x-x2)/(Xp-x2),2)-r3 - 4 * r3 * (x-x2)/hp * r5 * r4/TMath::Power((r4+xi),2);
  }

  if(TMath::Abs(r2) > 100){
    fit_result = 0;  
  }
  else{
    //---- Normalize the result
    fit_result = exp(r2);
  }
  
  return fit_result;
  
}
