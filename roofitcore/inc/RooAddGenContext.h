/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAddGenContext.rdl,v 1.11 2005/02/25 14:22:53 wverkerke Exp $
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
#ifndef ROO_ADD_GEN_CONTEXT
#define ROO_ADD_GEN_CONTEXT

#include "RooAbsGenContext.h"
#include "RooArgSet.h"

class RooAddPdf;
class RooDataSet;
class RooRealIntegral;
class RooAcceptReject;
class TRandom;
class TIterator;

class RooAddGenContext : public RooAbsGenContext {
public:
  RooAddGenContext(const RooAddPdf &model, const RooArgSet &vars, const RooDataSet *prototype= 0,
                   const RooArgSet* auxProto=0, Bool_t _verbose= kFALSE);
  virtual ~RooAddGenContext();

  virtual void setProtoDataOrder(Int_t* lut) ;

protected:

  virtual void initGenerator(const RooArgSet &theEvent);
  virtual void generateEvent(RooArgSet &theEvent, Int_t remaining);
  void updateThresholds() ;

  RooAddGenContext(const RooAddGenContext& other) ;

  const RooArgSet* _vars ;       
  RooArgSet* _pdfSet ;
  RooAddPdf *_pdf ;              //  Snapshot of PDF
  TList _gcList ;                //  List of component generator contexts
  Int_t  _nComp ;                //  Number of PDF components
  Double_t* _coefThresh ;        //[_nComp] Array of coefficient thresholds 

  ClassDef(RooAddGenContext,0) // Context for generating a dataset from a PDF
};

#endif
