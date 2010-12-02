 /***************************************************************************** 
  * Project: RooFit                                                           * 
  *                                                                           * 
  * Simple Poisson PDF
  * author: Kyle Cranmer <cranmer@cern.ch>
  *                                                                           * 
  *****************************************************************************/ 

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// Poisson pdf
// END_HTML
//

#include <iostream> 

#include "RooPoisson.h" 
#include "RooAbsReal.h" 
#include "RooAbsCategory.h" 

#include "RooRandom.h"
#include "RooMath.h"
#include "TMath.h"
#include "Math/ProbFuncMathCore.h"

ClassImp(RooPoisson) 



//_____________________________________________________________________________
RooPoisson::RooPoisson(const char *name, const char *title, 
		       RooAbsReal& _x,
		       RooAbsReal& _mean,
		       Bool_t noRounding) :
  RooAbsPdf(name,title), 
  x("x","x",this,_x),
  mean("mean","mean",this,_mean),
  _noRounding(noRounding)
{ 
  // Constructor
} 



//_____________________________________________________________________________
 RooPoisson::RooPoisson(const RooPoisson& other, const char* name) :  
   RooAbsPdf(other,name), 
   x("x",this,other.x),
   mean("mean",this,other.mean),
   _noRounding(other._noRounding)
{ 
   // Copy constructor
} 




//_____________________________________________________________________________
Double_t RooPoisson::evaluate() const 
{ 
  // Implementation in terms of the TMath Poisson function

  Double_t k = _noRounding ? x : floor(x);  
  return TMath::Poisson(k,mean) ;
} 





//_____________________________________________________________________________
Int_t RooPoisson::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const 
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  return 0 ;
}



//_____________________________________________________________________________
Double_t RooPoisson::analyticalIntegral(Int_t code, const char* rangeName) const 
{
  assert(code==1) ;

  // Implement integral over x as summation. Add special handling in case
  // range boundaries are not on integer values of x
  Double_t xmin = x.min(rangeName) ;
  Double_t xmax = x.max(rangeName) ;

  // Protect against negative lower boundaries
  if (xmin<0) xmin=0 ;

  Int_t ixmin = Int_t (xmin) ;
  Int_t ixmax = Int_t (xmax)+1 ;

  Double_t fracLoBin = 1-(xmin-ixmin) ;
  Double_t fracHiBin = 1-(ixmax-xmax) ;

  if (!x.hasMax()) {
    if (xmin<1e-6) {
      return 1 ;
    } else {
      
      // Return 1 minus integral from 0 to x.min() 

      if(ixmin == 0){ // first bin
	return TMath::Poisson(0, mean)*(xmin-0);
      }      
      Double_t sum(0) ;
      sum += TMath::Poisson(0,mean)*fracLoBin ;
      sum+= ROOT::Math::poisson_cdf(ixmin-2, mean) - ROOT::Math::poisson_cdf(0,mean) ;
      sum += TMath::Poisson(ixmin-1,mean)*fracHiBin ;
      return 1-sum ;
    }
  }
  
  if(ixmin == ixmax-1){ // first bin
    return TMath::Poisson(ixmin, mean)*(xmax-xmin);
  }  

  Double_t sum(0) ;
  sum += TMath::Poisson(ixmin,mean)*fracLoBin ;
  if (RooNumber::isInfinite(xmax)){
    sum+= 1.-ROOT::Math::poisson_cdf(ixmin,mean) ;
  }  else {
    sum+= ROOT::Math::poisson_cdf(ixmax-2, mean) - ROOT::Math::poisson_cdf(ixmin,mean) ;
    sum += TMath::Poisson(ixmax-1,mean)*fracHiBin ;
  }
  
  return sum ;

}







//_____________________________________________________________________________
Int_t RooPoisson::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t /*staticInitOK*/) const
{
  // Advertise internal generator in x

  if (matchArgs(directVars,generateVars,x)) return 1 ;  
  return 0 ;
}



//_____________________________________________________________________________
void RooPoisson::generateEvent(Int_t code)
{
  // Implement internal generator using TRandom::Poisson 

  assert(code==1) ;
  Double_t xgen ;
  while(1) {    
    xgen = RooRandom::randomGenerator()->Poisson(mean);
    if (xgen<=x.max() && xgen>=x.min()) {
      x = xgen ;
      break;
    }
  }
  return;
}


