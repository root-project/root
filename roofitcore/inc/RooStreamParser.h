/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_STREAM_PARSER
#define ROO_STREAM_PARSER

#include <iostream.h>
#include "TString.h"

class RooStreamParser {
public:
  // Constructors, assignment etc.
  RooStreamParser(istream& is) ;
  RooStreamParser(istream& is, TString errPrefix) ;
  virtual ~RooStreamParser();

  TString readToken() ;
  TString readLine() ;
  Bool_t expectToken(TString expected, Bool_t zapOnError=kFALSE) ;

  Bool_t readDouble(Double_t& value, Bool_t zapOnError=kFALSE) ;
  Bool_t convertToDouble(TString token, Double_t& value) ;

  Bool_t readInteger(Int_t value, Bool_t zapOnError=kFALSE) ;
  Bool_t convertToInteger(TString token, Int_t& value) ;

  Bool_t readString(TString& value, Bool_t zapOnError=kFALSE) ;
  Bool_t convertToString(TString token, TString& string) ;

  void zapToEnd() ;
  
protected:
  istream& _is ;
  TString _prefix ;
  ClassDef(RooStreamParser,0) // not persistable 
};

#endif
