/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooSuperCategory.h,v 1.17 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_SUPER_CATEGORY
#define ROO_SUPER_CATEGORY

class TObject ;
class TIterator;
#include "RooMultiCategory.h"
#include "RooAbsCategoryLValue.h"
#include "RooArgSet.h"
#include "RooTemplateProxy.h"


class RooSuperCategory : public RooAbsCategoryLValue {
public:
  // Constructors etc.
  RooSuperCategory();
  RooSuperCategory(const char *name, const char *title, const RooArgSet& inputCatList);
  RooSuperCategory(const RooSuperCategory& other, const char *name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooSuperCategory(*this,newname); }
  ~RooSuperCategory() override { };

  bool setIndex(value_type index, bool printError = true) override ;
  using RooAbsCategoryLValue::setIndex;
  bool setLabel(const char* label, bool printError=true) override;
  using RooAbsCategoryLValue::setLabel;

  // Printing interface (human readable)
  void printMultiline(std::ostream& os, Int_t content, bool verbose=false, TString indent="") const override;

  const RooArgSet& inputCatList() const { return _multiCat->inputCatList(); }

  bool inRange(const char* rangeName) const override;
  bool hasRange(const char* rangeName) const override;

protected:
  value_type evaluate() const override {
    return _multiCat->getCurrentIndex();
  }

  /// Ask server category to recompute shape, and copy its information.
  void recomputeShape() override {
    // Propagate up
    setShapeDirty();
    _multiCat->recomputeShape();
    _stateNames = _multiCat->_stateNames;
    _insertionOrder = _multiCat->_insertionOrder;
  }


private:
  RooTemplateProxy<RooMultiCategory> _multiCat;

  ClassDefOverride(RooSuperCategory,2) // Lvalue product operator for category lvalues
};

#endif
