/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooStreamParser.h,v 1.17 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_STREAM_PARSER
#define ROO_STREAM_PARSER

#include "TString.h"

class RooStreamParser {
public:
  // Constructors, assignment etc.
  RooStreamParser(std::istream& is) ;
  RooStreamParser(std::istream& is, const TString& errPrefix) ;
  virtual ~RooStreamParser();

  TString readToken() ;
  TString readLine() ;
  bool expectToken(const TString& expected, bool zapOnError=false) ;
  void setPunctuation(const TString& punct) ;
  TString getPunctuation() const { return _punct ; }

  bool readDouble(double& value, bool zapOnError=false) ;
  bool convertToDouble(const TString& token, double& value) ;

  bool readInteger(Int_t& value, bool zapOnError=false) ;
  bool convertToInteger(const TString& token, Int_t& value) ;

  bool readString(TString& value, bool zapOnError=false) ;
  bool convertToString(const TString& token, TString& string) ;

  bool atEOL() ;
  inline bool atEOF() { return _atEOF ; }
  void zapToEnd(bool inclContLines=false) ;

  bool isPunctChar(char c) const ;

protected:

  std::istream* _is ;
  bool _atEOL ;
  bool _atEOF ;
  TString _prefix ;
  TString _punct ;


  ClassDef(RooStreamParser,0) // Utility class that parses std::iostream data into tokens
};

#endif
