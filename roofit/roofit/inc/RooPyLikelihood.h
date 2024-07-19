#ifndef ROOPYLIKELIHOOD_H
#define ROOPYLIKELIHOOD_H

#include "RooAbsReal.h"
#include "RooListProxy.h"
#include "RooArgList.h"

class RooPyLikelihood : public RooAbsReal {
public:
   RooPyLikelihood(const char *name, const char *title, RooArgList &varlist);
   RooPyLikelihood(const RooPyLikelihood &right, const char *name = nullptr);
   virtual ~RooPyLikelihood();

   RooPyLikelihood* clone(const char *name) const override;
   Double_t evaluate() const override;
   const RooArgList& varlist() const;

protected:
   RooListProxy m_varlist; // all variables as list of variables

   ClassDefOverride(RooPyLikelihood, 1)
};

#endif


