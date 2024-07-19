#include "RooPyLikelihood.h"

ClassImp(RooPyLikelihood);

RooPyLikelihood::RooPyLikelihood(const char *name, const char *title, RooArgList &varlist) :
   RooAbsReal(name, title),
   m_varlist("!varlist", "All variables(list)", this) {
   m_varlist.add(varlist);
}

RooPyLikelihood::RooPyLikelihood(const RooPyLikelihood &right, const char *name) :
   RooAbsReal(right, name),
   m_varlist("!varlist", this, right.m_varlist) {
}

RooPyLikelihood::~RooPyLikelihood() {}

RooPyLikelihood* RooPyLikelihood::clone(const char *name) const {
   return new RooPyLikelihood(*this, name);
}

Double_t RooPyLikelihood::evaluate() const {
   // This function should be redefined in Python
   return 1;
}

const RooArgList& RooPyLikelihood::varlist() const {
   return m_varlist;
}
