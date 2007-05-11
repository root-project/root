/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooEffGenContext.cc,v 1.2 2005/06/23 07:37:30 wverkerke Exp $
 * Authors:                                                                  *
 *   GR, Gerhard Raven, NIKHEF/VU, Gerhard.Raven@nikhf.nl                    *
 *                                                                           *
 * Copyright (c) 2005, NIKHEF.  All rights reserved.                         *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
// -- CLASS DESCRIPTION [AUX] --
// A class description belongs here...
                                                                                                                      
                                                                                                                      
#include "RooFit.h"
#include "RooEffGenContext.h"
#include "RooAbsPdf.h"
#include "RooRandom.h"
using namespace std;

ClassImp(RooEffGenContext)
  ;

RooEffGenContext::RooEffGenContext(const RooAbsPdf &model, 
                 const RooAbsPdf& pdf, const RooAbsReal& eff,
                 const RooArgSet &vars,
                 const RooDataSet *prototype, const RooArgSet* auxProto,
                 Bool_t verbose, const RooArgSet* /*forceDirect*/) :
  RooAbsGenContext(model,vars,prototype,auxProto,verbose)
{
    RooArgSet x(eff,eff.GetName());
   _cloneSet = (RooArgSet*) x.snapshot(kTRUE);
   _eff = dynamic_cast<RooAbsReal*>(_cloneSet->find(eff.GetName()));
   _generator=pdf.genContext(vars,prototype,auxProto,verbose);
}


RooEffGenContext::~RooEffGenContext()
{
}

void RooEffGenContext::initGenerator(const RooArgSet &theEvent)
{
    _eff->recursiveRedirectServers(theEvent);
    _generator->initGenerator(theEvent);
}

void RooEffGenContext::generateEvent(RooArgSet &theEvent, Int_t remaining)
{
    Double_t maxEff=1; // for now -- later check max val of _eff...
    do {
        _generator->generateEvent(theEvent,remaining);
    } while (_eff->getVal() < RooRandom::uniform()*maxEff);
}
