/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsGenContext.h,v 1.15 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ABS_GEN_CONTEXT
#define ROO_ABS_GEN_CONTEXT

#include "TNamed.h"
#include "RooPrintable.h"
#include "RooArgSet.h"
#include "RooAbsPdf.h"

class RooDataSet;

class RooAbsGenContext : public TNamed, public RooPrintable {
public:
  RooAbsGenContext(const RooAbsPdf &model, const RooArgSet &vars, const RooDataSet *prototype= 0, const RooArgSet* auxProto=0,
		   Bool_t _verbose= kFALSE) ;
  virtual ~RooAbsGenContext();

  RooDataSet *generate(Int_t nEvents= 0);

  Bool_t isValid() const { return _isValid; }

  inline void setVerbose(Bool_t verbose= kTRUE) { _verbose= verbose; }
  inline Bool_t isVerbose() const { return _verbose; }

  virtual void setProtoDataOrder(Int_t* lut) ;

   inline virtual void Print(Option_t *options= 0) const {
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  virtual void attach(const RooArgSet& params) ;

  virtual void printName(ostream& os) const ;
  virtual void printTitle(ostream& os) const ;
  virtual void printClassName(ostream& os) const ;
  virtual void printArgs(ostream& os) const ;
  virtual void printMultiline(ostream& os, Int_t contents, Bool_t verbose=kFALSE, TString indent="") const;

  virtual Int_t defaultPrintContents(Option_t* opt) const ;
  virtual StyleOption defaultPrintStyle(Option_t* opt) const ;

protected:

  friend class RooConvGenContext ;
  friend class RooProdGenContext ;
  friend class RooAddGenContext ;
  friend class RooAddGenContextOrig ;
  friend class RooSimGenContext ;
  friend class RooEffGenContext ;

  virtual void initGenerator(const RooArgSet &theEvent);
  virtual void generateEvent(RooArgSet &theEvent, Int_t remaining) = 0;

  const RooDataSet *_prototype;
  RooArgSet *_theEvent;
  Bool_t _isValid;
  Bool_t _verbose;
  UInt_t _expectedEvents;
  RooArgSet _protoVars;
  Int_t _nextProtoIndex;
  RooAbsPdf::ExtendMode _extendMode ;
  Int_t* _protoOrder ;

  ClassDef(RooAbsGenContext,0) // Abstract context for generating a dataset from a PDF
};

#endif
