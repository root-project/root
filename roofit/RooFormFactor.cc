/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooFormFactor.cc,v 1.6 2001/08/03 18:13:02 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   MSG, Mandeep Gill, using RooGaussian framework,  extra Form Factor additions
 *
 * History:
 *   05-Jan-2000 DK Created initial version from RooGaussianProb
 *   02-May-2001 WV Port to RooFitModels/RooFitCore
 *   20-Aug-2001 MSG Begin adding FF stuff
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/
#include "BaBar/BaBar.hh"

#include <iostream.h>
#include <math.h>

#include "RooFitModels/RooFormFactor.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooRealVar.hh"

// #include "RooFitTools/RooRandom.hh"

// Need these next includes if i want to do any couts inside this code
// (but not for standalone macros in Root, because Root autoloads
// these things)

#include <stdio.h>
#include <iostream.h>

ClassImp(RooFormFactor)

static const char rcsid[] =
"$Id: RooFormFactor.cc,v 1.2 2001/05/23 22:10:04 msgill Exp $";

RooFormFactor::RooFormFactor(const char *name, const char *title,
			     RooAbsReal& _w, RooAbsReal& _ctl, RooAbsReal& _ctv, 
			     RooAbsReal& _chi,
			     RooAbsReal& _R1, RooAbsReal& _R2, RooAbsReal& _rho2) :

  // The two addresses refer to our first dependent variable and
  // parameter, respectively, as declared in the rdl file
  RooAbsPdf(name, title),

  // Declare our dependent variable(s) in the order they are listed
  // in the rdl file
  w("w"," w",this,_w),
  ctl("ctl"," ctl",this,_ctl),
  ctv("ctv"," ctv",this,_ctv),
  chi("chi"," chi",this,_chi),
  // Declare our parameter(s) in the order they are listed in the rdl file
  R1("R1"," R1",this,_R1),
  R2("R2"," R2",this,_R2),
  rho2("rho2"," rho2",this,_rho2)
{
}

// Copy ctor
RooFormFactor::RooFormFactor(const RooFormFactor& other,const  char *name):
			   RooAbsPdf(other,name)  ,
			   w("w",this,other.w),
			   ctl("ctl",this,other.ctl),
			   ctv("ctv",this,other.ctv),
			   chi("chi",this,other.chi),
			   R1("R1",this,other.R1),
			   R2("R2",this,other.R2),
			   rho2("rho2",this,other.rho2)
{
}

Int_t RooFormFactor::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars) const 
{
  if (matchArgs(allVars,analVars,w,ctl,ctv,chi)) return 1 ;
  return 0 ;
}



Double_t RooFormFactor::analyticalIntegral(Int_t code) const 
{
  switch(code) {
  case 0: return getVal() ; 
  case 1: 
    {

      Double_t ffnorm =  
	R1*R1*rho2*rho2*(33.064945  )+
	R1*R1*rho2*(-206.492447       )+
	R1*R1*( 370.425079          )+
	R1*rho2*rho2*( 113.676430     )+
	R1*rho2*(-714.117432           )+
	R1*(1259.422485                )+
	R2*R2*rho2*rho2*(222.014496    )+
	R2*R2*rho2*(-1091.489136        )+
	R2*R2*(1414.974976            )+
	R2*rho2*rho2*(-1122.696167      )+
	R2*rho2*(5701.543945           )+
	R2*( -7814.526855              )+
	R1*R2*rho2*rho2*(1.173239   )+
	R1*R2*rho2*(-5.295558        )+
	R1*R2*(  4.429727          )+
	rho2*rho2*(  1688.263306      )+
	rho2*(-9420.049805               )+
	(       15616.684570            ); 

      return ffnorm;
    }
  }
  
  assert(0) ;
  return 0 ;
}


Double_t RooFormFactor::evaluate(const RooArgSet* nset) const {
  
  // This is the 4-dim PDF depending on the wlvc vars, and with
  // R1,R2,rho2 as the independent params we'll want to fit for
  // (Neubert HQET FormFactor form)

  Double_t Pi=4*atan(1);
  
  Double_t mb=5.28;
  Double_t mdstr=2.01   ;
  
  Double_t mb2 = mb*mb;
  Double_t mdstr2 = mdstr*mdstr;

  Double_t stl = sqrt(1-ctl*ctl);
  Double_t stv = sqrt(1-ctv*ctv);
  Double_t cchi = cos(chi);
  Double_t c2chi = cos(2*chi);
  
  Double_t ctl2 = ctl*ctl;
  Double_t ctv2 = ctv*ctv;
  Double_t stl2 = stl*stl;
  Double_t stv2 = stv*stv;
 
  
  Double_t omctl2 = (1-ctl)*(1-ctl);
  Double_t opctl2 = (1+ctl)*(1+ctl);
  
  
  Double_t pdstr = mdstr*sqrt(w*w-1);
  
  Double_t opw2=(w+1)*(w+1);
  
  Double_t r=mdstr/mb ;
  Double_t rsq=r*r;
  Double_t omr2 = (1-r)*(1-r);
  
  
  Double_t ha1=(1-rho2*(w-1));
  
  Double_t hpfac = (1-sqrt( (w-1)/(w+1) ) * R1 );
  Double_t hmfac = (1+sqrt( (w-1)/(w+1) ) * R1 );
  Double_t hzfac = (1+((w-1)/(1-r))*(1-R2));
  
  Double_t hp= sqrt( (1-2*w*r+rsq)/omr2 )* hpfac;
  Double_t hm= sqrt( (1-2*w*r+rsq)/omr2 )* hmfac;
  Double_t hz= hzfac;  
	
  Double_t hphmterm = -2*hp*hm* stl*stl* stv*stv* c2chi;
  Double_t hphzterm = -4*hp*hz* stl*(1-ctl)*stv*ctv*cchi;
  Double_t hmhzterm = 4*hm*hz*stl*(1+ctl)*stv*ctv*cchi;
    
  Double_t hp2term = hp*hp* stv*stv * omctl2 ;
  Double_t hm2term = hm*hm* stv*stv* opctl2;
  Double_t hz2term = hz*hz* 4* stl2* ctv2;
    
  Double_t dgd4= (ha1*ha1)* opw2 * pdstr* 
    (hp2term + hm2term + hz2term + hphmterm + hphzterm + hmhzterm);

  //assert(0) ; // ?
  return dgd4;
}
