// -- CLASS DESCRIPTION [PDF] --
// Landau Distribution PDF...

#include "RooFitModels/RooLandau.hh"
#include "RooFitCore/RooRandom.hh"

ClassImp(RooLandau)

RooLandau::RooLandau(const char *name, const char *title, RooAbsReal& _x, RooAbsReal& _mean, RooAbsReal& _sigma) :
  RooAbsPdf(name,title),
  x("x","Dependent",this,_x),
  mean("mean","Mean",this,_mean),
  sigma("sigma","Width",this,_sigma)
{
}
 
RooLandau::RooLandau(const RooLandau& other, const char* name) : 
  RooAbsPdf(other,name),
  x("x",this,other.x),
  mean("mean",this,other.mean),
  sigma("sigma",this,other.sigma)
{
} 

Double_t RooLandau::evaluate() const
{
  return TMath::Landau(x, mean, sigma);
}

Int_t RooLandau::getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK) const
{
  if (matchArgs(directVars,generateVars,x)) return 1 ;  
  return 0 ;
}

void RooLandau::generateEvent(Int_t code)
{
  assert(code==1) ;
  Double_t xgen ;
  while(1) {    
    xgen = RooRandom::randomGenerator()->Landau(mean,sigma);
    if (xgen<x.max() && xgen>x.min()) {
      x = xgen ;
      break;
    }
  }
  return;
}


