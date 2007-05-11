/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSimGenContext.rdl,v 1.11 2005/02/25 14:23:02 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_SIM_GEN_CONTEXT
#define ROO_SIM_GEN_CONTEXT

#include "TList.h"
#include "RooAbsGenContext.h"
#include "RooArgSet.h"

class RooSimultaneous;
class RooDataSet;
class RooAbsCategoryLValue ;

class RooSimGenContext : public RooAbsGenContext {
public:
  RooSimGenContext(const RooSimultaneous &model, const RooArgSet &vars, const RooDataSet *prototype= 0,
                   const RooArgSet* auxProto=0, Bool_t _verbose= kFALSE);
  virtual ~RooSimGenContext();
  virtual void setProtoDataOrder(Int_t* lut) ;

protected:

  virtual void initGenerator(const RooArgSet &theEvent);
  virtual void generateEvent(RooArgSet &theEvent, Int_t remaining);

  RooSimGenContext(const RooSimGenContext& other) ;

  RooAbsCategoryLValue* _idxCat ; // Clone of index category
  RooArgSet*            _idxCatSet ; // Owner of index category components
  const RooDataSet *_prototype;   // Prototype data set
  const RooSimultaneous *_pdf ;   // Original PDF
  TList _gcList ;                 // List of component generator contexts
  Bool_t _haveIdxProto ;          // Flag set if generation of index is requested
  TString _idxCatName ;           // Name of index category
  Int_t _numPdf ;                 // Number of generated PDFs
  Double_t* _fracThresh ;         //[_numPdf] Fraction threshold array

  ClassDef(RooSimGenContext,0) // Context for generating a dataset from a PDF
};

#endif
