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

#include "TError.h"

using namespace std;

ClassImp(RooPoisson) 



//_____________________________________________________________________________
RooPoisson::RooPoisson(const char *name, const char *title, 
		       RooAbsReal& _x,
		       RooAbsReal& _mean,
		       Bool_t noRounding) :
  RooAbsPdf(name,title), 
  x("x","x",this,_x),
  mean("mean","mean",this,_mean),
  _noRounding(noRounding),
  _protectNegative(false)
{ 
  // Constructor  
} 



//_____________________________________________________________________________
 RooPoisson::RooPoisson(const RooPoisson& other, const char* name) :  
   RooAbsPdf(other,name), 
   x("x",this,other.x),
   mean("mean",this,other.mean),
   _noRounding(other._noRounding),
   _protectNegative(other._protectNegative)
{ 
   // Copy constructor
} 




//_____________________________________________________________________________
Double_t RooPoisson::evaluate() const 
{ 
  // Implementation in terms of the TMath Poisson function

  Double_t k = _noRounding ? x : floor(x);  
  if(_protectNegative && mean<0) 
    return 1e-3;
  return TMath::Poisson(k,mean) ;
} 



//_____________________________________________________________________________
Double_t RooPoisson::getLogVal(const RooArgSet* s) const 
{
  // calculate and return the negative log-likelihood of the Poisson                                                                                                                                    
  return RooAbsPdf::getLogVal(s) ;
//   Double_t prob = getVal(s) ;
//   return prob ;

  // Make inputs to naming conventions of RooAbsPdf::extendedTerm
  Double_t expected=mean ;
  Double_t observed=x ;

  // Explicitly handle case Nobs=Nexp=0
  if (fabs(expected)<1e-10 && fabs(observed)<1e-10) {
    return 0 ;
  }  

  // Explicitly handle case Nexp=0
  if (fabs(observed)<1e-10) {
    return -1*expected;
  }

  // Michaels code for log(poisson) in RooAbsPdf::extendedTer with an approximated log(observed!) term
  Double_t extra=0;
  if(observed<1000000) {
    extra = -observed*log(expected)+expected+TMath::LnGamma(observed+1.);    
  } else {
    //if many observed events, use Gauss approximation                                                                                                                                                 
    Double_t sigma_square=expected;
    Double_t diff=observed-expected;
    extra=-log(sigma_square)/2 + (diff*diff)/(2*sigma_square);
  }

//   if (fabs(extra)>100 || log(prob)>100) {
//     cout << "RooPoisson::getLogVal(" << GetName() << ") mu=" << expected << " x = " << x << " -log(P) = " << extra << " log(evaluate()) = " << log(prob) << endl ;
//   }
  
//   if (fabs(extra+log(prob))>1) {
//     cout << "RooPoisson::getLogVal(" << GetName() << ") WARNING mu=" << expected << " x = " << x << " -log(P) = " << extra << " log(evaluate()) = " << log(prob) << endl ;
//   }

  //return log(prob);
  return -extra-analyticalIntegral(1,0) ; //log(prob);

}


//_____________________________________________________________________________
Int_t RooPoisson::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* /*rangeName*/) const 
{
  if (matchArgs(allVars,analVars,x)) return 1 ;
  if (matchArgs(allVars, analVars, mean)) return 2;
  return 0 ;
}



//_____________________________________________________________________________
Double_t RooPoisson::analyticalIntegral(Int_t code, const char* rangeName) const 
{
  R__ASSERT(code == 1 || code == 2) ;

  if(_protectNegative && mean<0) 
    return exp(-2*mean); // make it fall quickly

  if (code == 1) {
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

  } else if(code == 2) {
   
    // the integral with respect to the mean is the integral of a gamma distribution 
    Double_t mean_min = mean.min(rangeName);
    Double_t mean_max = mean.max(rangeName);
     
    Double_t ix;
    if(_noRounding) ix = x + 1;
    else ix = Int_t(TMath::Floor(x)) + 1.0; // negative ix does not need protection (gamma returns 0.0)
    
    return ROOT::Math::gamma_cdf(mean_max, ix, 1.0) - ROOT::Math::gamma_cdf(mean_min, ix, 1.0);
  }

  return 0;

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

  R__ASSERT(code==1) ;
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


