/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooProduct.rdl,v 1.1 2003/04/28 20:42:41 wverkerke Exp $
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
#ifndef ROO_PRODUCT
#define ROO_PRODUCT

#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooSetProxy.hh"

class RooRealVar;
class RooArgList ;

class RooProduct : public RooAbsReal {
public:

  RooProduct() ;
  RooProduct(const char *name, const char *title, const RooArgSet& _prodSet) ;

  RooProduct(const RooProduct& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooProduct(*this, newname); }
  inline virtual ~RooProduct() { }

protected:

  RooSetProxy _compSet ;
  TIterator* _compIter ;  //! do not persist

  Double_t evaluate() const;

  ClassDef(RooProduct,1) // Product of RooAbsReal terms
};

#endif
