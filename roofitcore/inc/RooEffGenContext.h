/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooEffGenContext.rdl,v 1.1 2005/06/20 15:44:51 wverkerke Exp $
 * Authors:                                                                  *
 *   GR, Gerhard Raven, NIKHEF/VU, Gerhard.Raven@nikhf.nl                    *
 *                                                                           *
 * Copyright (c) 2005, NIKHEF.  All rights reserved.                         *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_EFF_GEN_CONTEXT
#define ROO_EFF_GEN_CONTEXT
                                                                                                                      
#include "RooAbsGenContext.h"
class RooAbsPdf;
class RooArgSet;
class RooDataSet;
class RooAbsReal;

class RooEffGenContext : public RooAbsGenContext {
public:
  RooEffGenContext(const RooAbsPdf &model, 
                   const RooAbsPdf &pdf,const RooAbsReal& eff,
                   const RooArgSet &vars, const RooDataSet *prototype= 0,
                   const RooArgSet* auxProto=0, Bool_t verbose=kFALSE, const RooArgSet* forceDirect=0);
  virtual ~RooEffGenContext();

protected:
  void initGenerator(const RooArgSet &theEvent);
  void generateEvent(RooArgSet &theEvent, Int_t remaining);
private:
   RooArgSet *_cloneSet;
   RooAbsReal *_eff;
   RooAbsGenContext *_generator;

   ClassDef(RooEffGenContext,0) // Context for generating a dataset from a PDF
};
#endif
