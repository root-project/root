/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooThresholdCategory.h,v 1.8 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_THRESHOLD_CATEGORY
#define ROO_THRESHOLD_CATEGORY

#include "RooAbsCategory.h"
#include "RooRealProxy.h"
#include <vector>
#include <utility>

#include "RooFitLegacy/RooCatTypeLegacy.h"

class RooThresholdCategory : public RooAbsCategory {

public:
  // Constructors etc.
  RooThresholdCategory() {};
  RooThresholdCategory(const char *name, const char *title, RooAbsReal& inputVar,
      const char* defCatName="Default", Int_t defCatIdx=0);
  RooThresholdCategory(const RooThresholdCategory& other, const char *name=0) ;
  TObject* clone(const char* newname) const override { return new RooThresholdCategory(*this, newname); }

  // Mapping function
  bool addThreshold(Double_t upperLimit, const char* catName, Int_t catIdx=-99999) ;

  // Printing interface (human readable)
  void printMultiline(std::ostream& os, Int_t content, bool verbose=false, TString indent="") const override ;

  void writeToStream(std::ostream& os, bool compact) const override ;

protected:

  RooRealProxy _inputVar ;
  const value_type _defIndex{std::numeric_limits<value_type>::min()};
  std::vector<std::pair<double,value_type>> _threshList;

  value_type evaluate() const override ;
  /// No shape recomputation is necessary. This category does not depend on other categories.
  void recomputeShape() override { }

  ClassDefOverride(RooThresholdCategory, 3) // Real-to-Category function defined by series of thresholds
};

#endif
