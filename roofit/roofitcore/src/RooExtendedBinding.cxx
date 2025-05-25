/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooExtendedBinding.h>


RooExtendedBinding::RooExtendedBinding(const char *name, const char *title, RooAbsPdf &_pdf)
   : RooAbsReal(name, title), pdf("pdf", "pdf", this, _pdf)
{
}

RooExtendedBinding::RooExtendedBinding(const char *name, const char *title, RooAbsPdf &_pdf, const RooArgSet &obs)
   : RooExtendedBinding{name, title, _pdf}
{
   _obsList = std::make_unique<RooSetProxy>("obsList", "List of observables", this, false, false);
   _obsList->add(obs);
}

RooExtendedBinding::RooExtendedBinding(const RooExtendedBinding &other, const char *name)
   : RooAbsReal(other, name), pdf("pdf", this, other.pdf)
{
   if (other._obsList) {
      _obsList = std::make_unique<RooSetProxy>("obsList", this, *other._obsList);
   }
}

double RooExtendedBinding::evaluate() const
{
   RooArgSet const *normSet = _obsList ? _obsList.get() : nullptr;
   return (const_cast<RooAbsPdf &>(static_cast<RooAbsPdf const &>(pdf.arg()))).expectedEvents(normSet);
}
