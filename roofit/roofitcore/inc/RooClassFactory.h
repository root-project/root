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
#include "RooFactoryWSTool.h"

#include <vector>
#include <string>

class RooAbsReal;
class RooAbsPdf;

class RooClassFactory : public TNamed, public RooPrintable {

public:

  // Constructors, assignment etc
  RooClassFactory() ;
  ~RooClassFactory() override ;

  static RooAbsReal* makeFunctionInstance(const char* className, const char* name, const char* expression, const RooArgList& vars, const char* intExpression=0) ;
  static RooAbsReal* makeFunctionInstance(const char* name, const char* expression, const RooArgList& vars, const char* intExpression=0) ;

  static RooAbsPdf* makePdfInstance(const char* className, const char* name, const char* expression, const RooArgList& vars, const char* intExpression=0) ;
  static RooAbsPdf* makePdfInstance(const char* name, const char* expression, const RooArgList& vars, const char* intExpression=0) ;

  static bool makeAndCompilePdf(const char* name, const char* expression, const RooArgList& vars, const char* intExpression=0) ;
  static bool makeAndCompileFunction(const char* name, const char* expression, const RooArgList& args, const char* intExpression=0) ;

  static bool makePdf(const char* name, const char* realArgNames=0, const char* catArgNames=0,
         const char* expression="1.0", bool hasAnaInt=false, bool hasIntGen=false, const char* intExpression=0) ;
  static bool makeFunction(const char* name, const char* realArgNames=0, const char* catArgNames=0,
              const char* expression="1.0", bool hasAnaInt=false, const char* intExpression=0) ;
  static bool makeClass(const char* className, const char* name, const char* realArgNames=0, const char* catArgNames=0,
           const char* expression="1.0", bool hasAnaInt=false, bool hasIntGen=false, const char* intExpression=0) ;

  class ClassFacIFace : public RooFactoryWSTool::IFace {
  public:
    std::string create(RooFactoryWSTool& ft, const char* typeName, const char* instanceName, std::vector<std::string> args) override ;
  } ;

protected:



  RooClassFactory(const RooClassFactory&) ;

  ClassDefOverride(RooClassFactory,0) // RooFit class code and instance factory
} ;

#endif
