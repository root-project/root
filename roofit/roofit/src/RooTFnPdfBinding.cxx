/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/** \class RooTFnPdfBinding
    \ingroup Roofit

**/

#include "Riostream.h"

#include "RooTFnPdfBinding.h"
#include "RooAbsCategory.h"
#include "TF3.h"

using std::ostream;


////////////////////////////////////////////////////////////////////////////////

RooTFnPdfBinding::RooTFnPdfBinding(const char *name, const char *title, TF1* func, const RooArgList& list) :
  RooAbsPdf(name,title),
  _list("params","params",this),
  _func(func)
{
  _list.add(list) ;
}

////////////////////////////////////////////////////////////////////////////////

RooTFnPdfBinding::RooTFnPdfBinding(const RooTFnPdfBinding& other, const char* name) :
  RooAbsPdf(other,name),
  _list("params",this,other._list),
  _func(other._func)
{
}

////////////////////////////////////////////////////////////////////////////////

double RooTFnPdfBinding::evaluate() const
{
  double x = _list.at(0) ? (static_cast<RooAbsReal*>(_list.at(0)))->getVal() : 0 ;
  double y = _list.at(1) ? (static_cast<RooAbsReal*>(_list.at(1)))->getVal() : 0 ;
  double z = _list.at(2) ? (static_cast<RooAbsReal*>(_list.at(2)))->getVal() : 0 ;
  return _func->Eval(x,y,z) ;
}

////////////////////////////////////////////////////////////////////////////////

void RooTFnPdfBinding::printArgs(ostream& os) const
{
  // Print object arguments and name/address of function pointer
  os << "[ TFn={" << _func->GetName() << "=" << _func->GetTitle() << "} " ;
  for (Int_t i=0 ; i<numProxies() ; i++) {
    RooAbsProxy* p = getProxy(i) ;
    if (!TString(p->name()).BeginsWith("!")) {
      p->print(os) ;
      os << " " ;
    }
  }
  os << "]" ;
}

////////////////////////////////////////////////////////////////////////////////

namespace RooFit {

  RooAbsPdf* bindPdf(TF1* func,RooAbsReal& x) {
    return new RooTFnPdfBinding(func->GetName(),func->GetName(),func,x) ;
  }

  RooAbsPdf* bindPdf(TF2* func,RooAbsReal& x, RooAbsReal& y) {
    return new RooTFnPdfBinding(func->GetName(),func->GetName(),func,RooArgList(x,y)) ;
  }

  RooAbsPdf* bindPdf(TF3* func,RooAbsReal& x, RooAbsReal& y, RooAbsReal& z) {
    return new RooTFnPdfBinding(func->GetName(),func->GetName(),func,RooArgList(x,y,z)) ;
  }

}
