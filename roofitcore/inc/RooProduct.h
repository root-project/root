/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooProduct.rdl,v 1.4 2005/12/01 16:10:20 wverkerke Exp $
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
#ifndef ROO_PRODUCT
#define ROO_PRODUCT

#include "RooAbsReal.h"
#include "RooSetProxy.h"

class RooRealVar;
class RooArgList ;

class RooProduct : public RooAbsReal {
public:

  RooProduct() ;
  RooProduct(const char *name, const char *title, const RooArgSet& _prodSet) ;

  RooProduct(const RooProduct& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooProduct(*this, newname); }
  virtual ~RooProduct() ;

protected:

  RooSetProxy _compRSet ;
  RooSetProxy _compCSet ;
  TIterator* _compRIter ;  //! do not persist
  TIterator* _compCIter ;  //! do not persist

  Double_t evaluate() const;

  ClassDef(RooProduct,1) // Product of RooAbsReal and RooAbsCategory terms
};

#endif
