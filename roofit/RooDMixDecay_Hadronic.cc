/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooDMixDecay_Hadronic.cc,v 1.1 2002/03/28 02:29:38 mwilson Exp $
 * Authors:
 *   MW, Michael Wilson, UC Santa Cruz, mwilson@slac.stanford.edu
 * History:
 *   08-Mar-2002 MW Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
//

#include "RooFitModels/RooDMixDecay_Hadronic.hh"

#include <cmath>
#include "RooFitCore/RooRealIntegral.hh"
#include "RooFitCore/RooRandom.hh"

ClassImp(RooDMixDecay_Hadronic)


RooDMixDecay_Hadronic::RooDMixDecay_Hadronic(const char *name, const char *title,
						     RooRealVar& t_, RooAbsReal& tau_,
						     RooAbsCategory& flavor_, RooAbsReal& R_DCS_,
						     RooAbsReal& yprime_, RooAbsReal& xprime_,
						     RooAbsReal& A_D_, RooAbsReal& A_M_,
						     RooAbsReal& phi_,
						     const RooResolutionModel& model_) :
  RooConvolutedPdf(name, title, model_, t_),
  t("t","Decay Time",this,t_),
  tau("tau","Decay Lifetime",this,tau_),
  yprime("yprime","Mixing Parameter y",this,yprime_),
  xprime("xprime","Mixing Parameter x^2",this,xprime_),
  R_DCS("R_DCS","Doubly-Cabbibo-Supressed Decay Rate",this,R_DCS_),
  A_D("A_D","A_D Decay CP Violation",this,A_D_),
  A_M("A_M","A_M Mixing CP Violation",this,A_M_),
  phi("phi","phi Interference CP Violation",this,phi_),
  flavor("flavor","D0 flavor",this,flavor_)
{

  // Define which basis functions to use in our model

  basisConst = declareBasis("exp(-@0/@1)",RooArgList(tau_));
  basisLin   = declareBasis("(@0/@1)*exp(-@0/@1)",RooArgList(tau_));
  basisQuad  = declareBasis("(@0/@1)*(@0/@1)*exp(-@0/@1)",RooArgList(tau_));

}


RooDMixDecay_Hadronic::RooDMixDecay_Hadronic(const RooDMixDecay_Hadronic& other,
						     const char *name) :
  RooConvolutedPdf(other,name),
  flavor("flavor",this,other.flavor),
  t("t",this,other.t),
  tau("tau",this,other.tau),
  yprime("yprime",this,other.yprime),
  xprime("xprime",this,other.xprime),
  R_DCS("R_DCS",this,other.R_DCS),
  A_D("A_D",this,other.A_D),
  A_M("A_M",this,other.A_M),
  phi("phi",this,other.phi),
  basisConst(other.basisConst),
  basisLin(other.basisLin),
  basisQuad(other.basisQuad)
{
  // Copy constructor
}


RooDMixDecay_Hadronic::~RooDMixDecay_Hadronic()
{
  // Destructor
}



Double_t RooDMixDecay_Hadronic::coefficient(Int_t basisIndex) const
{
  if(basisIndex==basisConst) {
    return 1.0;
  }

  Double_t R_M2 = sqrt( (1+flavor*A_M)/(1-flavor*A_M) );
  Double_t R_D2 = sqrt( (1-flavor*A_D)/(1+flavor*A_D) );

  if(basisIndex==basisLin) {
    Double_t yP;

    if(phi != 0) {
      yP = yprime*cos(phi) - flavor*xprime*sin(phi);
    } else {
      yP = yprime;
    }

    return sqrt(R_M2*R_D2/R_DCS) * yP;
  }

  if(basisIndex==basisQuad) {

    return ( (R_M2*R_D2)/(4.0*R_DCS) * (xprime*xprime + yprime*yprime) );
  }

  cerr << "Unknown basisIndex " << basisIndex << endl;
  assert(0);
  return 0.0;
}


Int_t RooDMixDecay_Hadronic::getCoefAnalyticalIntegral(RooArgSet& allVars,
							   RooArgSet& analVars) const
{
  if (matchArgs(allVars,analVars,flavor)) return 1;
  
  return 0;
}

Double_t RooDMixDecay_Hadronic::coefAnalyticalIntegral(Int_t coef, Int_t code) const
{

  switch(code) {
    //No integration
  case 0:
    return coefficient(coef);
    
  case 1:
    if(coef==basisConst) {
      return 2.0;
    }
    
    Double_t R_M2_D0 = sqrt( (1+A_M)/(1-A_M) );
    Double_t R_D2_D0 = sqrt( (1-A_D)/(1+A_D) );
    
    if(coef==basisLin) {
      Double_t yP, xP;
      
      if(phi != 0) {
        yP = yprime*cos(phi);
        xP = xprime*sin(phi);
      } else {
        yP = yprime;
        xP = 0.0;
      }
      
      Double_t f = sqrt(R_M2_D0*R_D2_D0);

      return sqrt(1.0/R_DCS)*( yP*(1.0/f + f) + xP*(1.0/f - f) );
    }
    
    if(coef==basisQuad) {
      Double_t f2 = R_M2_D0*R_D2_D0;
      
      return ( 1.0/(4.0*R_DCS) * (xprime*xprime + yprime*yprime) * (1.0/f2 + f2) );
    }
    
  }

  cerr << "Illegal code in coefAnalyticIntegral: " << code << endl;
  assert(0);
  return 0.0;
}


Int_t RooDMixDecay_Hadronic::getGenerator(const RooArgSet& directVars,
					      RooArgSet& generateVars) const
{
  if (matchArgs(directVars,generateVars,t,flavor))  return 2;
  if (matchArgs(directVars,generateVars,t)) return 1;

  return 0;
}


void RooDMixDecay_Hadronic::initGenerator(Int_t code)
{
  switch (code) {
  case 2:

    // Calculate the fraction of D0bar events to generate
    Double_t sumInt =
      RooRealIntegral("sumInt","sum integral",*this,RooArgSet(t.arg(),flavor.arg())).getVal();
    flavor = D0;
    Double_t flavInt = RooRealIntegral("flavInt","flavor integral",*this,RooArgSet(t.arg())).getVal();
    genFlavorFrac = flavInt/sumInt;

    break;
  }

  // The PDF we have defined in this class is actually based on a Taylor expansion, and it
  // is not correct as t -> inf.  Here we define the upper bound on where we will generate
  // values of t.  The expansion is from ~sin( yprime*t/tau ) ( and ~cos( xprime*t/tau ) ), and
  // by requiring that yprime*(t/tau) < 1 (not necessarily a good assumption, but real data
  // may have this), we get t < tau/yprime.  For yprime = 0.1, this means t < 10*tau.

  genMaxT = 10*tau;

  // The maximum function, which is supposed to be greater than the PDF at all points for which
  // we generate t values, is defined as 
  // 
  //      genMaxCoeff * exp(-t/genMaxLife)
  //

  Double_t R_M2(0);
  Double_t R_D2(0);

  if(A_M < A_D) {
    R_M2 = sqrt( (1-A_M)/(1+A_M) );
    R_D2 = sqrt( (1+A_D)/(1-A_D) );
  }
  else {
    R_M2 = sqrt( (1+A_M)/(1-A_M) );
    R_D2 = sqrt( (1-A_D)/(1+A_D) );
  }

  Double_t yP(0);

  if(phi != 0) {
    yP = yprime*cos(phi) + fabs(xprime*sin(phi));
  } else {
    yP = yprime;
  }

  // In here, we find a good shape for the exponential by fitting to two points.  The point
  // at t/tau = 1 is chosen in case the linear term is positive, in which case the PDF
  // will bend upwards slightly and we want our exponential to be above it always.  The
  // point at t = genMaxT is chosen because the quadratic term dies off more slowly than
  // a pure exponential.

  // Finally, we make sure that the leading coefficient is not less than 1.0

  Double_t scale = R_M2 * R_D2 / R_DCS;
  Double_t C1 = (scale/4.0) * (xprime*xprime + yprime*yprime) * 4.0;    // max at t/tau = 2
  Double_t C2 = sqrt(scale) * yP * 1.0;                                 // max at t/tau = 1
  Double_t C3 = 1.0;

  Double_t y1 = (C1 + C2 + C3)*exp(-1.0);                               // eval at t/tau = 1
  Double_t t1 = tau;

  Double_t D1 = (scale/4.0) * (xprime*xprime + yprime*yprime) * (genMaxT/tau)*(genMaxT/tau);
  Double_t D2 = sqrt(scale) * yP * (genMaxT/tau);
  Double_t D3 = 1.0;

  Double_t y2 = (D1 + D2 + D3)*exp(-genMaxT/tau);
  Double_t t2 = genMaxT;

  genMaxLife = (t2-t1)/(log(y1/y2));
  genMaxCoeff = y1*exp(t1/genMaxLife);
  if(genMaxCoeff < 1.0) genMaxCoeff = 1.0;

  // extra paranoia, just in case the exponential is very close (or identical)
  // to the PDF

  genMaxCoeff *= 1.1;

  genMaxArea = genMaxCoeff * genMaxLife * (1 - exp(-genMaxT/genMaxLife));

}


void RooDMixDecay_Hadronic::generateEvent(Int_t code)
{

  // Generate D0 flavor
  switch(code) {
  case 2:
    Double_t rand = RooRandom::uniform();
    flavor = (Int_t) ((rand<=genFlavorFrac) ? D0 : D0bar) ;
    break;
  }

  // Generate t dependent
  try {
    while(1) {
      Double_t rand = genMaxArea * RooRandom::uniform();
      Double_t tval = -genMaxLife * log(1 - rand/(genMaxCoeff*genMaxLife));
      
      Double_t maxProb = genMaxCoeff*exp(-tval/genMaxLife);

      Double_t prob = maxProb * RooRandom::uniform();
      
      Double_t R_M2 = sqrt( (1+flavor*A_M)/(1-flavor*A_M) );
      Double_t R_D2 = sqrt( (1-flavor*A_D)/(1+flavor*A_D) );
      
      Double_t yP(0);
      
      if(phi != 0) {
	yP = yprime*cos(phi) - flavor*xprime*sin(phi);
      } else {
	yP = yprime;
      }
      
      Double_t scale = R_M2 * R_D2 / R_DCS;
      Double_t tprime = tval/tau;
      
      Double_t acceptProb =
	exp(-tprime) * ( 1.0 + sqrt(scale) * yP * tprime +
			 (scale/4.0) * (xprime*xprime + yprime*yprime) * (tprime*tprime));
      
      // If the maximum probability is not always greated than the PDF, we
      // have a problem.
      
      if(maxProb < acceptProb) throw MaxProbError(maxProb,acceptProb,tval);
      
      Bool_t accept = (prob < acceptProb) ? kTRUE : kFALSE;
      
      if (tval<t.max() && tval>t.min() && accept) {
	t = tval;
	break;
      }
    }
  }
  catch(MaxProbError err) {
    cerr << "maximum probability function is less than PDF:" << endl;
    cerr << "  maxProb = "    << err.max << endl;
    cerr << "  acceptProb = " << err.accept << endl;
    cerr << "  tval = "       << err.tval << endl;
    cerr << "Exiting now"     << endl;
    exit(1);
  }
}
