 /***************************************************************************** 
  * Project: RooFit                                                           * 
  *                                                                           * 
  * Simple Gamma distribution 
  * authors: Stefan A. Schmitz, Gregory Schott
  *                                                                           * 
  *****************************************************************************/ 

//
// implementation of the Gamma PDF for RooFit/RooStats
// $f(x) = \frac{(x-\mu)^{\gamma-1} \cdot \exp^{(-(x-mu) / \beta}}{\Gamma(\gamma) \cdot \beta^{\gamma}}$
// defined for $x \geq 0$ if $\mu = 0$
//

// Notes from Kyle Cranmer
// Wikipedia and several sources refer to the Gamma distribution as
// G(mu|alpha,beta) = beta^alpha mu^(alpha-1) e^(-beta mu) / Gamma(alpha)
// Below is the correspondance
// Wikipedia | This Implementation
//---------------------------------
// alpha     | gamma
// beta      | 1/beta
// mu        | x
// 0         | mu
//
// Note, that for a model Pois(N|mu), a uniform prior on mu, and a measurement N
// the posterior is in the Wikipedia parametrization Gamma(mu, alpha=N+1, beta=1)
// thus for this implementation it is:
// RooGamma(_x=mu,_gamma=N+1,_beta=1,_mu=0)
// Also note, that in this case it is equivalent to
// RooPoison(N,mu) and treating the function as a PDF in mu.


#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "RooGamma.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooRandom.h"
#include "RooMath.h"

#include <iostream> 
#include "TMath.h"

#include <Math/SpecFuncMathCore.h>
#include <Math/PdfFuncMathCore.h>
#include <Math/ProbFuncMathCore.h>

#include "TError.h"

using namespace std;

ClassImp(RooGamma)


//_____________________________________________________________________________
RooGamma::RooGamma(const char *name, const char *title,
			 RooAbsReal& _x, RooAbsReal& _gamma,
			 RooAbsReal& _beta, RooAbsReal& _mu) :
  RooAbsPdf(name,title),
  x("x","Observable",this,_x),
  gamma("gamma","Mean",this,_gamma),
  beta("beta","Width",this,_beta),
  mu("mu","Para",this,_mu)
{
}



//_____________________________________________________________________________
RooGamma::RooGamma(const RooGamma& other, const char* name) : 
  RooAbsPdf(other,name), x("x",this,other.x), gamma("mean",this,other.gamma),
  beta("beta",this,other.beta), mu("mu",this,other.mu)
{
}



//_____________________________________________________________________________
Double_t RooGamma::evaluate() const
{

  Double_t arg= x ;  
  Double_t ret = TMath::GammaDist(arg, gamma, mu, beta) ;
  return ret ;
}



//_____________________________________________________________________________
Int_t RooGamma::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const 
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  return 0 ;
}



//_____________________________________________________________________________
Double_t RooGamma::analyticalIntegral(Int_t code, const char* rangeName) const 
{
  R__ASSERT(code==1) ;

 //integral of the Gamma distribution via ROOT::Math
  Double_t integral = ROOT::Math::gamma_cdf(x.max(rangeName), gamma, beta, mu) - ROOT::Math::gamma_cdf(x.min(rangeName), gamma, beta, mu);
  return integral ;


}




//_____________________________________________________________________________
Int_t RooGamma::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t /*staticInitOK*/) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ;  
  return 0 ;
}



//_____________________________________________________________________________
void RooGamma::generateEvent(Int_t code)
{
  R__ASSERT(code==1) ;
//algorithm adapted from code example in:
//Marsaglia, G. and Tsang, W. W.
//A Simple Method for Generating Gamma Variables
//ACM Transactions on Mathematical Software, Vol. 26, No. 3, September 2000
//
//The speed of this algorithm depends on the speed of generating normal variates.
//The algorithm is limited to $\gamma \geq 0$ !

  while(1) {

  double d = 0;
  double c = 0;
  double xgen = 0;
  double v = 0;
  double u = 0; 
  d = gamma -1./3.; c = 1./TMath::Sqrt(9.*d);

  while(v <= 0.){
    xgen = RooRandom::randomGenerator()->Gaus(); v = 1. + c*xgen;
  }
  v = v*v*v; u = RooRandom::randomGenerator()->Uniform();
  if( u < 1.-.0331*(xgen*xgen)*(xgen*xgen) ) {
    if ( (((d*v)* beta + mu ) < x.max()) && (((d*v)* beta + mu) > x.min()) ) {
      x = ((d*v)* beta + mu) ;
      break;
    } 
  } 
  if( TMath::Log(u) < 0.5*xgen*xgen + d*(1.-v + TMath::Log(v)) ) { 
    if ( (((d*v)* beta + mu ) < x.max()) && (((d*v)* beta + mu) > x.min()) ) {
      x = ((d*v)* beta + mu) ;
      break;
    } 
  } 

  }


  return;
}


