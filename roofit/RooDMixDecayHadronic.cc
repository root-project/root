/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitModels
 *    File: $Id: RooDMixDecayHadronic.cc,v 1.1 2002/03/29 21:24:27 mwilson Exp $
 * Authors:
 *   MW, Michael Wilson, UC Santa Cruz, mwilson@slac.stanford.edu
 * History:
 *   08-Mar-2002 MW Created initial version
 *
 * Copyright (C) 2002 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [PDF] --
//

#include "RooFitModels/RooDMixDecayHadronic.hh"

#include <cmath>
#include "RooFitCore/RooRealIntegral.hh"
#include "RooFitCore/RooRandom.hh"

ClassImp(RooDMixDecayHadronic)


RooDMixDecayHadronic::RooDMixDecayHadronic(const char *name, const char *title,
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
  //
  // Constructor
  //

  // Define which basis functions to use in our model

  basisConst = declareBasis("exp(-@0/@1)",RooArgList(tau_));
  basisLin   = declareBasis("(@0/@1)*exp(-@0/@1)",RooArgList(tau_));
  basisQuad  = declareBasis("(@0/@1)*(@0/@1)*exp(-@0/@1)",RooArgList(tau_));

}


RooDMixDecayHadronic::RooDMixDecayHadronic(const RooDMixDecayHadronic& other,
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
  //
  // Copy constructor
  //
}


RooDMixDecayHadronic::~RooDMixDecayHadronic()
{
  //
  // Destructor
  //
}



Double_t RooDMixDecayHadronic::coefficient(Int_t basisIndex) const
{
  //
  // Returns the coefficients for the basis functions used in the PDF definition
  //

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

  std::cerr << "Unknown basisIndex " << basisIndex << std::endl;
  assert(0);
  return 0.0;
}


Int_t RooDMixDecayHadronic::getCoefAnalyticalIntegral(RooArgSet& allVars,
							   RooArgSet& analVars) const
{
  //
  // Returns a code for the analytic integral of the coefficients:
  //
  //    returns 1 if we need to integrate over the D0 flavor
  //    returns 0 otherwise (no integration)

  if (matchArgs(allVars,analVars,flavor)) return 1;
  
  return 0;
}

Double_t RooDMixDecayHadronic::coefAnalyticalIntegral(Int_t coef, Int_t code) const
{
  //
  // Returns the analytic integral of the coefficient based on the code 
  // from getCoefAnalyticalIntegral()

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

  std::cerr << "Illegal code in coefAnalyticIntegral: " << code << std::endl;
  assert(0);
  return 0.0;
}


Int_t RooDMixDecayHadronic::getGenerator(const RooArgSet& directVars,
					      RooArgSet& generateVars) const
{
  //
  // Returns a code for the ToyMC generator needed:
  //
  //    returns 2 if we need to generate t and flavor dependents
  //    returns 1 if we need to generate t
  //    returns 0 otherwise

  if (matchArgs(directVars,generateVars,t,flavor))  return 2;
  if (matchArgs(directVars,generateVars,t)) return 1;

  return 0;
}


void RooDMixDecayHadronic::initGenerator(Int_t code)
{
  //
  // Called at the beginning of ToyMC generation to initialize the generator

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

  // In here, we find a good shape for the generating exponential by calculating the PDF
  // value and the value of its derivative at a point, and the exponential to have the
  // same value and derivative at that point.  A good point to choose is near t/tau = 2,
  // because this is where the t^2 term has its maximum, and the t term is still significant.
  // The places where we run into danger are between 1 < t/tau < 2, and for t/tau -> inf.


  Double_t scale = R_M2 * R_D2 / R_DCS;
  Double_t tTerm = sqrt(scale) * yP;
  Double_t t2Term = (scale/4.0)*(xprime*xprime+yprime*yprime);

  Double_t evalPt = 2;    // we evaluate at t/tau = 2

  Double_t func = exp(-evalPt)*(1.0 + tTerm*(evalPt) + t2Term*(evalPt)*(evalPt));
  Double_t deriv = (1/tau)*exp(-evalPt)*(tTerm - 1.0 + (2.0*t2Term - tTerm)*(evalPt)
					      - t2Term*(evalPt)*(evalPt));

  genMaxLife = -func/deriv;
  genMaxCoeff = func*exp(5.0*tau/genMaxLife);

  // sanity check, make sure that the coefficient is not too small!

  genMaxCoeff *= 1.05;
  if(genMaxCoeff < 1.05) genMaxCoeff = 1.05;

  genMaxArea = genMaxCoeff * genMaxLife * (1 - exp(-genMaxT/genMaxLife));

}


void RooDMixDecayHadronic::generateEvent(Int_t code)
{
  //
  // The is the function which implements the ToyMC generator.

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

      if(maxProb < acceptProb) throw MaxProbError(maxProb,acceptProb,tval,sqrt(scale)*yP,
						  (scale/4.0)*(xprime*xprime+yprime*yprime));
      
      Bool_t accept = (prob < acceptProb) ? kTRUE : kFALSE;
      
      if (tval<t.max() && tval>t.min() && accept) {
	t = tval;
	break;
      }
    }
  }
  catch(MaxProbError err) {
    std::cerr << "maximum probability function is less than PDF:" << std::endl;
    std::cerr << "  maxProb = "    << err.max << "\t";
    std::cerr << "acceptProb = " << err.accept << "\t";
    std::cerr << "tval = "       << err.tval << std::endl;
    std::cerr << "  genMaxLife = " << genMaxLife << "\t";
    std::cerr << "genMaxCoeff = " << genMaxCoeff << std::endl;
    std::cerr << "  tau = " << tau << "\t";
    std::cerr << "t term = " << err.tTerm << "\t";
    std::cerr << "t^2 term = " << err.t2Term << std::endl;
    std::cerr << "Exiting now"     << std::endl;
    exit(1);
  }
}
