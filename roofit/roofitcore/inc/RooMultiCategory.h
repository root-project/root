/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooMultiCategory.h,v 1.9 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_MULTI_CATEGORY
#define ROO_MULTI_CATEGORY

class TObject;
#include "RooAbsCategoryLValue.h"
#include "RooArgSet.h"
#include "RooSetProxy.h"
#include <string>

class RooSuperCategory;


class RooMultiCategory : public RooAbsCategory {
public:
  // Constructors etc.
  inline RooMultiCategory() { setShapeDirty(); }
  RooMultiCategory(const char *name, const char *title, const RooArgSet& inputCatList);
  RooMultiCategory(const RooMultiCategory& other, const char *name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooMultiCategory(*this,newname); }
  ~RooMultiCategory() override;

  // Printing interface (human readable)
  void printMultiline(std::ostream& os, Int_t content, bool verbose=false, TString indent="") const override;

  /// Multi categories cannot be read from streams.
  bool readFromStream(std::istream& /*is*/, bool /*compact*/, bool /*verbose=false*/) override { return true; }
  void writeToStream(std::ostream& os, bool compact) const override;

  const RooArgSet& inputCatList() const { return _catSet ; }
  const char* getCurrentLabel() const override;

protected:

  std::string createLabel() const;
  value_type evaluate() const override;
  void recomputeShape() override;

  RooSetProxy _catSet ; ///< Set of input category

  friend class RooSuperCategory;
  ClassDefOverride(RooMultiCategory,1) // Product operator for categories
};

#endif
