/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooStreamParser.rdl,v 1.1 2001/03/19 15:57:32 verkerke Exp $
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
  void putBackToken(TString& token) ;

  Bool_t readDouble(Double_t& value, Bool_t zapOnError=kFALSE) ;
  Bool_t convertToDouble(TString token, Double_t& value) ;

  Bool_t readInteger(Int_t value, Bool_t zapOnError=kFALSE) ;
  Bool_t convertToInteger(TString token, Int_t& value) ;

  Bool_t readString(TString& value, Bool_t zapOnError=kFALSE) ;
  Bool_t convertToString(TString token, TString& string) ;

  inline Bool_t atEOL() { return (_is.peek()=='\n') ; }
  void zapToEnd() ;
  
protected:
  istream& _is ;
  Bool_t _atEOL ;
  TString _prefix ;
  ClassDef(RooStreamParser,0) // not persistable 
};

#endif
