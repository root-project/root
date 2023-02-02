// @(#)root/foam:$Id$
// Author: S. Jadach <mailto:Stanislaw.jadach@ifj.edu.pl>, P.Sawicki <mailto:Pawel.Sawicki@ifj.edu.pl>

#ifndef ROOT_TFoamIntegrand
#define ROOT_TFoamIntegrand

#include "TObject.h"

class TFoamIntegrand : public TObject  {
public:
   TFoamIntegrand() { };
   ~TFoamIntegrand() override { };
   virtual Double_t Density(Int_t ndim, Double_t *) = 0;

   ClassDefOverride(TFoamIntegrand,1); //n-dimensional real positive integrand of FOAM
};

#endif
