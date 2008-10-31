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


//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// RooEffGenContext is a specialized generator context for p.d.fs represented
// by class RooEffProd, which are p.d.fs multiplied with an efficiency function.
// This generator context generates events from such products by first
// generating events from a dedicated generator context of the input p.d.f.
// and applying an extra rejection step based on the efficiency function.
// END_HTML
//
                                                                                                                      
                                                                                                                      
#include "RooFit.h"
#include "RooEffGenContext.h"
#include "RooAbsPdf.h"
#include "RooRandom.h"
using namespace std;

ClassImp(RooEffGenContext)
  ;


//_____________________________________________________________________________
RooEffGenContext::RooEffGenContext(const RooAbsPdf &model, 
                 const RooAbsPdf& pdf, const RooAbsReal& eff,
                 const RooArgSet &vars,
                 const RooDataSet *prototype, const RooArgSet* auxProto,
                 Bool_t verbose, const RooArgSet* /*forceDirect*/) :
  RooAbsGenContext(model,vars,prototype,auxProto,verbose)
{
  // Constructor of generator context for RooEffProd products

    RooArgSet x(eff,eff.GetName());
   _cloneSet = (RooArgSet*) x.snapshot(kTRUE);
   _eff = dynamic_cast<RooAbsReal*>(_cloneSet->find(eff.GetName()));
   _generator=pdf.genContext(vars,prototype,auxProto,verbose);
}



//_____________________________________________________________________________
RooEffGenContext::~RooEffGenContext()
{
  // Destructor
  delete _generator ;
  delete _cloneSet ;
}


//_____________________________________________________________________________
void RooEffGenContext::initGenerator(const RooArgSet &theEvent)
{
  // One-time initialization of generator.

    _eff->recursiveRedirectServers(theEvent);
    _generator->initGenerator(theEvent);
}


//_____________________________________________________________________________
void RooEffGenContext::generateEvent(RooArgSet &theEvent, Int_t remaining)
{
  // Generate one event. Generate an event from the p.d.f and
  // then perform an accept/reject sampling based on the efficiency
  // function

    Double_t maxEff=1; // for now -- later check max val of _eff...
    do {
        _generator->generateEvent(theEvent,remaining);
    } while (_eff->getVal() < RooRandom::uniform()*maxEff);
}
