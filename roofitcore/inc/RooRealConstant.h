/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealConstant.rdl,v 1.3 2002/05/03 21:49:56 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 * History:
 *   16-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_REAL_CONSTANT
#define ROO_REAL_CONSTANT

#include "Rtypes.h"

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

