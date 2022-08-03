/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsCategoryLValue.h,v 1.22 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_ABS_CATEGORY_LVALUE
#define ROO_ABS_CATEGORY_LVALUE

#include "RooAbsCategory.h"
#include "RooAbsLValue.h"
#include <list>
#include <string>
#include <utility>

class RooAbsCategoryLValue : public RooAbsCategory, public RooAbsLValue {
public:
  // Constructor, assignment etc.
  RooAbsCategoryLValue() {
    // Default constructor
  } ;
  RooAbsCategoryLValue(const char *name, const char *title);
  RooAbsCategoryLValue(const RooAbsCategoryLValue& other, const char* name=nullptr) ;
  ~RooAbsCategoryLValue() override;

  // Value modifiers
  ////////////////////////////////////////////////////////////////////////////////
  /// Change category state by specifying the index code of the desired state.
  /// If printError is set, a message will be printed if
  /// the specified index does not represent a valid state.
  /// \return bool to signal an error.
  virtual bool setIndex(value_type index, bool printError = true) = 0;
  ////////////////////////////////////////////////////////////////////////////////
  /// Change category state to state specified by another category state.
  /// If printError is set, a message will be printed if
  /// the specified index does not represent a valid state.
  /// \note The state name of the other category is ignored.
  /// \return bool to signal an error.
  bool setIndex(const std::pair<std::string,value_type>& nameIdxPair, bool printError = true) {
    return setIndex(nameIdxPair.second, printError);
  }
  bool setOrdinal(unsigned int index);

  ////////////////////////////////////////////////////////////////////////////////
  /// Change category state by specifying a state name.
  /// If printError is set, a message will be printed if
  /// the specified state name does not represent a valid state.
  /// \return bool to signal an error.
  virtual bool setLabel(const char* label, bool printError=true) = 0;
  /// \copydoc setLabel(const char*, bool)
  bool setLabel(const std::string& label, bool printError = true) {
    return setLabel(label.c_str(), printError);
  }
  ////////////////////////////////////////////////////////////////////////////////
  /// Change category state to the state name of another category.
  /// If printError is set, a message will be printed if
  /// the specified state name does not represent a valid state.
  /// \note The state index of the other category is ignored.
  /// \return bool to signal an error.
  bool setLabel(const std::pair<std::string,value_type>& nameIdxPair, bool printError = true) {
    return setLabel(nameIdxPair.first.c_str(), printError);
  }


  RooAbsArg& operator=(int index) ;
  RooAbsArg& operator=(const char* label) ;
  RooAbsArg& operator=(const RooAbsCategory& other) ;

  // Binned fit interface
  void setBin(Int_t ibin, const char* rangeName=nullptr) override ;
  /// Get the index of the plot bin for the current value of this category.
  Int_t getBin(const char* /*rangeName*/=nullptr) const override {
    return getCurrentOrdinalNumber();
  }
  Int_t numBins(const char* rangeName=nullptr) const override ;
  double getBinWidth(Int_t /*i*/, const char* /*rangeName*/=nullptr) const override {
    // Return volume of i-th bin (according to binning named rangeName if rangeName!=nullptr)
    return 1.0 ;
  }
  double volume(const char* rangeName) const override {
    // Return span of range with given name (=number of states included in this range)
    return numTypes(rangeName) ;
  }
  void randomize(const char* rangeName=nullptr) override;

  const RooAbsBinning* getBinningPtr(const char* /*rangeName*/) const override { return 0 ; }
  std::list<std::string> getBinningNames() const override { return std::list<std::string>(1, "") ; }
  Int_t getBin(const RooAbsBinning* /*ptr*/) const override { return getBin((const char*)0) ; }


  inline void setConstant(bool value= true) {
    // Declare category constant
    setAttribute("Constant",value);
  }

  inline bool isLValue() const override {
    // Object is an l-value
    return true;
  }

protected:

  friend class RooSimGenContext ;
  friend class RooSimSplitGenContext ;
  /// \cond
  /// \deprecated This function is useless. Use setIndex() instead.
  virtual void setIndexFast(Int_t index) {
    _currentIndex = index;
  }
  /// \endcond

  void copyCache(const RooAbsArg* source, bool valueOnly=false, bool setValDirty=true) override ;

  ClassDefOverride(RooAbsCategoryLValue,1) // Abstract modifiable index variable
};

#endif
