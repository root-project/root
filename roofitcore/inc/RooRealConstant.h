/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealConstant.rdl,v 1.7 2004/04/05 22:44:12 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_REAL_CONSTANT
#define ROO_REAL_CONSTANT

#include "Rtypes.h"
#include <iostream>

class RooAbsReal ;
class RooArgList ;
class TIterator ;
#include "RooFitCore/RooConstVar.hh"

class RooRealConstant {
public:

  static RooConstVar& value(Double_t value) ;

protected:

  static void init() ;

  static RooArgList* _constDB ;    // List of already instantiated constants
  static TIterator* _constDBIter ; // Iterator over constants list

  ClassDef(RooRealConstant,0) // RooRealVar constants factory
};

RooConstVar& RooConst(Double_t val) ; 


#endif

