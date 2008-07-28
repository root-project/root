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


ClassImp(RooPoisson) 



//_____________________________________________________________________________
RooPoisson::RooPoisson(const char *name, const char *title, 
		       RooAbsReal& _x,
		       RooAbsReal& _mean) :
  RooAbsPdf(name,title), 
  x("x","x",this,_x),
  mean("mean","mean",this,_mean)
{ 
  // Constructor
} 



//_____________________________________________________________________________
 RooPoisson::RooPoisson(const RooPoisson& other, const char* name) :  
   RooAbsPdf(other,name), 
   x("x",this,other.x),
   mean("mean",this,other.mean)
{ 
   // Copy constructor
} 




//_____________________________________________________________________________
Double_t RooPoisson::evaluate() const 
{ 
  // Implementation in terms of the TMath Poisson function

  Double_t k = floor(x);  
  return TMath::Poisson(k,mean) ;
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
    if (xgen<x.max() && xgen>x.min()) {
      x = xgen ;
      break;
    }
  }
  return;
}


