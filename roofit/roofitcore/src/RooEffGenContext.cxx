/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   GR, Gerhard Raven, NIKHEF/VU, Gerhard.Raven@nikhf.nl                    *
 *                                                                           *
 * Copyright (c) 2005, NIKHEF.  All rights reserved.                         *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/


/**
\file RooEffGenContext.cxx
\class RooEffGenContext
\ingroup Roofitcore

RooEffGenContext is a specialized generator context for p.d.fs represented
by class RooEffProd, which are p.d.fs multiplied with an efficiency function.
This generator context generates events from such products by first
generating events from a dedicated generator context of the input p.d.f.
and applying an extra rejection step based on the efficiency function.
**/

#include <memory>

#include "RooEffGenContext.h"
#include "RooAbsPdf.h"
#include "RooRandom.h"

using namespace std;

ClassImp(RooEffGenContext);

////////////////////////////////////////////////////////////////////////////////
/// Constructor of generator context for RooEffProd products

RooEffGenContext::RooEffGenContext(const RooAbsPdf &model,
                                   const RooAbsPdf& pdf, const RooAbsReal& eff,
                                   const RooArgSet &vars,
                                   const RooDataSet *prototype, const RooArgSet* auxProto,
                                   bool verbose, const RooArgSet* /*forceDirect*/) :
   RooAbsGenContext(model, vars, prototype, auxProto, verbose), _maxEff(0.)
{
   RooArgSet x(eff,eff.GetName());
   _cloneSet = static_cast<RooArgSet*>(x.snapshot(true));
   _eff = dynamic_cast<RooAbsReal*>(_cloneSet->find(eff.GetName()));
   _generator = pdf.genContext(vars, prototype, auxProto, verbose);
   _vars = static_cast<RooArgSet*>(vars.snapshot(true));
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooEffGenContext::~RooEffGenContext()
{
   delete _generator;
   delete _cloneSet;
   delete _vars;
}

////////////////////////////////////////////////////////////////////////////////
/// One-time initialization of generator.

void RooEffGenContext::initGenerator(const RooArgSet &theEvent)
{
   _eff->recursiveRedirectServers(theEvent);
   _generator->initGenerator(theEvent);

   // Check if PDF supports maximum finding
   Int_t code = _eff->getMaxVal(*_vars);
   if (!code) {
     _maxEff = 1.;
   } else {
     _maxEff = _eff->maxVal(code);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Generate one event. Generate an event from the p.d.f and
/// then perform an accept/reject sampling based on the efficiency
/// function

void RooEffGenContext::generateEvent(RooArgSet &theEvent, Int_t remaining)
{
   while (true) {
      _generator->generateEvent(theEvent, remaining);
      double val = _eff->getVal();
      if (val > _maxEff && !_eff->getMaxVal(*_vars)) {
         coutE(Generation) << ClassName() << "::" << GetName()
              << ":generateEvent: value of efficiency is larger than assumed maximum of 1."  << std::endl;
         continue;
      }
      if (val > RooRandom::uniform() * _maxEff) {
         break;
      }
   }
}
