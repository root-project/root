// $Id: TFoamIntegrand.h,v 1.1 2005/04/01 11:41:34 psawicki Exp $

#ifndef ROOT_TFoamIntegrand
#define ROOT_TFoamIntegrand

//_________________________________________
// Class TFoamIntegrand
// =====================
// Abstract class representing n-dimensional real positive integrand function

#include "TROOT.h"
class TFoamIntegrand : public TObject  {
 public:
  TFoamIntegrand() { };
  virtual ~TFoamIntegrand() { };
  virtual Double_t Density(Int_t ndim, Double_t *) = 0;

  ClassDef(TFoamIntegrand,1); //n-dimensional real positive integrand of FOAM
};
#endif
