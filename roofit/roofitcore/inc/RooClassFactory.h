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

#ifndef RooFit_RooClassFactory_h
#define RooFit_RooClassFactory_h

#include <string>

class RooAbsReal;
class RooAbsPdf;
class RooArgList;

// RooFit class code and instance factory
class RooClassFactory {

public:
   static RooAbsReal *makeFunctionInstance(std::string const &className, std::string const &name,
                                           std::string const &expression, const RooArgList &vars,
                                           std::string const &intExpression = "");
   static RooAbsReal *makeFunctionInstance(std::string const &name, std::string const &expression,
                                           const RooArgList &vars, std::string const &intExpression = "");

   static RooAbsPdf *makePdfInstance(std::string const &className, std::string const &name,
                                     std::string const &expression, const RooArgList &vars,
                                     std::string const &intExpression = "");
   static RooAbsPdf *makePdfInstance(std::string const &name, std::string const &expression, const RooArgList &vars,
                                     std::string const &intExpression = "");

   static bool makeAndCompilePdf(std::string const &name, std::string const &expression, const RooArgList &vars,
                                 std::string const &intExpression = "");
   static bool makeAndCompileFunction(std::string const &name, std::string const &expression, const RooArgList &args,
                                      std::string const &intExpression = "");

   static bool makePdf(std::string const &name, std::string const &realArgNames = "",
                       std::string const &catArgNames = "", std::string const &expression = "1.0",
                       bool hasAnaInt = false, bool hasIntGen = false, std::string const &intExpression = "");
   static bool makeFunction(std::string const &name, std::string const &realArgNames = "",
                            std::string const &catArgNames = "", std::string const &expression = "1.0",
                            bool hasAnaInt = false, std::string const &intExpression = "");
   static bool makeClass(std::string const &baseName, const std::string &className,
                         std::string const &realArgNames = "", std::string const &catArgNames = "",
                         std::string const &expression = "1.0", bool hasAnaInt = false, bool hasIntGen = false,
                         std::string const &intExpression = "");
};

#endif
