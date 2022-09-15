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
  RooAbsGenContext(const RooAbsPdf &model, const RooArgSet &vars, const RooDataSet *prototype= nullptr, const RooArgSet* auxProto=nullptr,
         bool _verbose= false) ;

  virtual RooDataSet *generate(double nEvents= 0, bool skipInit=false, bool extendedMode=false);

  bool isValid() const {
    // If true generator context is in a valid state
    return _isValid;
  }

  inline void setVerbose(bool verbose= true) {
    // Set/clear verbose messaging
    _verbose= verbose;
  }
  inline bool isVerbose() const {
    // If true verbose messaging is active
    return _verbose;
  }

  virtual void setProtoDataOrder(Int_t* lut) ;

   inline void Print(Option_t *options= nullptr) const override {
     // Print context information on stdout
     printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

  virtual void attach(const RooArgSet& params) ;

  void printName(std::ostream& os) const override ;
  void printTitle(std::ostream& os) const override ;
  void printClassName(std::ostream& os) const override ;
  void printArgs(std::ostream& os) const override ;
  void printMultiline(std::ostream& os, Int_t contents, bool verbose=false, TString indent="") const override;

  Int_t defaultPrintContents(Option_t* opt) const override ;
  StyleOption defaultPrintStyle(Option_t* opt) const override ;

  virtual void setExpectedData(bool) {} ;

  virtual void generateEvent(RooArgSet &theEvent, Int_t remaining) = 0;
  virtual void initGenerator(const RooArgSet &theEvent);

protected:

  virtual RooDataSet* createDataSet(const char* name, const char* title, const RooArgSet& obs) ;

  void resampleData(double& ratio) ;

  const RooDataSet *_prototype; ///< Pointer to prototype dataset
  RooArgSet _theEvent;          ///< Pointer to observable event being generated
  bool _isValid;              ///< Is context in valid state?
  bool _verbose;              ///< Verbose messaging?
  UInt_t _expectedEvents;       ///< Number of expected events from extended p.d.f
  RooArgSet _protoVars;         ///< Prototype observables
  Int_t _nextProtoIndex;        ///< Next prototype event to load according to LUT
  RooAbsPdf::ExtendMode _extendMode ;  ///< Extended mode capabilities of p.d.f.
  std::vector<Int_t> _protoOrder ; ///< LUT with traversal order of prototype data
  TString _normRange ;          ///< Normalization range of pdf

  RooDataSet* _genData ;        ///<! Data being generated

  ClassDefOverride(RooAbsGenContext,0) // Abstract context for generating a dataset from a PDF
};

#endif
