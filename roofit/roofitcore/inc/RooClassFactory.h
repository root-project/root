/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooClassFactory.h,v 1.2 2007/05/11 09:11:30 verkerke Exp $
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

#ifndef ROO_CODE_FACTORY
#define ROO_CODE_FACTORY

#include "TNamed.h"
#include "RooArgSet.h"
#include "RooPrintable.h"

#include <vector>
#include <string>

class RooAbsReal;
class RooAbsPdf;

// RooFit class code and instance factory
class RooClassFactory {

public:

  // Constructors, assignment etc
  RooClassFactory() ;
  ~RooClassFactory() ;

  static RooAbsReal* makeFunctionInstance(const char* className, const char* name, const char* expression, const RooArgList& vars, const char* intExpression=nullptr) ;
  static RooAbsReal* makeFunctionInstance(const char* name, const char* expression, const RooArgList& vars, const char* intExpression=nullptr) ;

  static RooAbsPdf* makePdfInstance(const char* className, const char* name, const char* expression, const RooArgList& vars, const char* intExpression=nullptr) ;
  static RooAbsPdf* makePdfInstance(const char* name, const char* expression, const RooArgList& vars, const char* intExpression=nullptr) ;

  static bool makeAndCompilePdf(const char* name, const char* expression, const RooArgList& vars, const char* intExpression=nullptr) ;
  static bool makeAndCompileFunction(const char* name, const char* expression, const RooArgList& args, const char* intExpression=nullptr) ;

  static bool makePdf(const char* name, const char* realArgNames=nullptr, const char* catArgNames=nullptr,
         const char* expression="1.0", bool hasAnaInt=false, bool hasIntGen=false, const char* intExpression=nullptr) ;
  static bool makeFunction(const char* name, const char* realArgNames=nullptr, const char* catArgNames=nullptr,
              const char* expression="1.0", bool hasAnaInt=false, const char* intExpression=nullptr) ;
  static bool makeClass(const char* className, const char* name, const char* realArgNames=nullptr, const char* catArgNames=nullptr,
           const char* expression="1.0", bool hasAnaInt=false, bool hasIntGen=false, const char* intExpression=nullptr) ;

protected:



  RooClassFactory(const RooClassFactory&) ;
} ;

#endif
