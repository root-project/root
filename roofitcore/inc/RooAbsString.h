/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsString.rdl,v 1.4 2001/05/03 02:15:54 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   26-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ABS_STRING
#define ROO_ABS_STRING

#include "TObjString.h"
#include "RooFitCore/RooAbsArg.hh"
class RooArgSet ;
class TH1F ;

class RooAbsString : public RooAbsArg {
public:

  // Constructors, assignment etc
  inline RooAbsString() { }
  RooAbsString(const char *name, const char *title) ;
  RooAbsString(const RooAbsString& other, const char* name=0);
  virtual ~RooAbsString();

  // Return value and unit accessors
  virtual TString getVal() const ;
  Bool_t operator==(TString value) const ;

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent="") const ;

protected:

  // Function evaluation and error tracing
  TString traceEval() const ;
  virtual Bool_t traceEvalHook(TString value) const { return kFALSE ; }
  virtual TString evaluate() const { return 0 ; }

  // Internal consistency checking (needed by RooDataSet)
  virtual Bool_t isValid() const ;
  virtual Bool_t isValid(TString value) const ;

  virtual void syncCache(const RooDataSet* dset=0) { getVal() ; }
  void copyCache(const RooAbsArg* source) ;
  virtual void attachToTree(TTree& t, Int_t bufSize=32000) ;

  mutable char _value[1024] ;

  ClassDef(RooAbsString,1) // a real-valued variable and its value
};

#endif
