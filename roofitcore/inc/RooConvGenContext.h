/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooConvGenContext.rdl,v 1.11 2005/12/08 13:19:54 wverkerke Exp $
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
#ifndef ROO_CONV_GEN_CONTEXT
#define ROO_CONV_GEN_CONTEXT

#include "RooAbsGenContext.h"
#include "RooArgSet.h"

class RooAbsAnaConvPdf;
class RooDataSet;
class RooRealIntegral;
class RooAcceptReject;
class TRandom;
class TIterator;
class RooRealVar ;
class RooNumConvPdf ;

class RooConvGenContext : public RooAbsGenContext {
public:
  RooConvGenContext(const RooNumConvPdf &model, const RooArgSet &vars, const RooDataSet *prototype= 0,
		    const RooArgSet* auxProto=0, Bool_t _verbose= kFALSE);
  RooConvGenContext(const RooAbsAnaConvPdf &model, const RooArgSet &vars, const RooDataSet *prototype= 0,
		    const RooArgSet* auxProto=0, Bool_t _verbose= kFALSE);
  virtual ~RooConvGenContext();

  virtual void setProtoDataOrder(Int_t* lut) ;

  virtual void printToStream(ostream &os, PrintOption opt= Standard, TString indent= "") const ;


protected:

  virtual void initGenerator(const RooArgSet &theEvent);
  virtual void generateEvent(RooArgSet &theEvent, Int_t remaining);

  RooConvGenContext(const RooConvGenContext& other) ;

  RooAbsGenContext* _pdfGen ;   // Physics model generator context
  RooAbsGenContext* _modelGen ; // Resolution model generator context
  TString _convVarName ;        // Name of convolution variable
  RooArgSet* _pdfVars ;         // Holder of PDF x truth event
  RooArgSet* _modelVars ;       // Holder of resModel event
  RooArgSet* _pdfCloneSet ;     // Owner of PDF clone
  RooArgSet* _modelCloneSet ;   // Owner of resModel clone
  RooRealVar* _cvModel ;         // Convolution variable in resModel event
  RooRealVar* _cvPdf ;           // Convolution variable in PDFxTruth event
  RooRealVar* _cvOut ;           // Convolution variable in output event

  ClassDef(RooConvGenContext,0) // Context for generating a dataset from a PDF
};

#endif
