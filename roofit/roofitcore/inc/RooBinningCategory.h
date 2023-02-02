/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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
#ifndef ROO_BINNING_CATEGORY
#define ROO_BINNING_CATEGORY

#include "RooAbsCategory.h"
#include "RooTemplateProxy.h"
#include "TString.h"

class RooBinningCategory : public RooAbsCategory {

public:
  // Constructors etc.
  inline RooBinningCategory() { }
  RooBinningCategory(const char *name, const char *title, RooAbsRealLValue& inputVar, const char* binningName=nullptr, const char* catTypeName=nullptr);
  RooBinningCategory(const RooBinningCategory& other, const char *name=nullptr) ;
  TObject* clone(const char* newname) const override { return new RooBinningCategory(*this, newname); }
  ~RooBinningCategory() override;

  /// Printing interface (human readable)
  void printMultiline(std::ostream& os, Int_t content, bool verbose=false, TString indent="") const override ;

protected:

  void initialize(const char* catTypeName=nullptr) ;

  RooTemplateProxy<RooAbsRealLValue> _inputVar; ///< Input variable that is mapped
  TString _bname ;         ///< Name of the binning specification to be used to perform the mapping

  value_type evaluate() const override;
  /// The shape of this category does not need to be recomputed, as it creates states on the fly.
  void recomputeShape() override { }

  ClassDefOverride(RooBinningCategory,1) // RealVar-to-Category function defined by bin boundaries on input var
};

#endif
