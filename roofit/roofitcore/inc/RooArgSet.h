/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooArgSet.h,v 1.45 2007/08/09 19:55:47 wouter Exp $
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
#ifndef ROO_ARG_SET
#define ROO_ARG_SET

#include "RooAbsCollection.h"
#include "RooAbsArg.h"

class RooArgList ;


#define USEMEMPOOLFORARGSET
template <class RooSet_t, size_t>
class MemPoolForRooSets;

class RooArgSet : public RooAbsCollection {
public:
  
#ifdef USEMEMPOOLFORARGSET
  void* operator new (size_t bytes);
  void* operator new (size_t bytes, void* ptr) noexcept;
  void operator delete (void *ptr);
#endif
 
  // Constructors, assignment etc.
  RooArgSet();
  RooArgSet(const RooArgList& list, const RooAbsArg* var1) ;
  explicit RooArgSet(const TCollection& tcoll, const char* name="") ;
  explicit RooArgSet(const char *name);

private:
  // Helper to check the types of the variadic template below:
  template<bool...> struct bool_pack;
  template<bool... bs>
  using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;
public:
  /// Construct a (non-owning) RooArgSet from one or more
  /// RooFit objects or collections. The set will not own its contents.
  /// \tparam T A RooFit class that derives from RooAbsArg or a RooFit collection.
  /// \tparam Ts Parameter pack of objects that derive from RooAbsArg or RooFit collections; or a name.
  /// \param arg A RooFit object that derives from RooAbsArg, or a collection of RooFit objects.
  /// \param moreArgsOrName Arbitrary number of
  /// - Further RooFit objects that derive from RooAbsArg
  /// - RooFit collections of such objects
  /// - A name for the set. Given multiple names, the last-given name prevails.
  template<typename T, typename... Ts,
      typename = typename std::enable_if<std::is_base_of<RooAbsArg, T>::value || std::is_base_of<RooAbsCollection, T>::value, void>::type,
      typename = typename std::enable_if<all_true<
      std::is_base_of<RooAbsArg, Ts>::value
      || std::is_base_of<RooAbsCollection, Ts>::value
      || std::is_convertible<Ts, const char*>::value ...>::value, void>::type >
  RooArgSet(const T& arg, const Ts&... moreArgsOrName)
  /* NB: Making this a delegating constructor lead to linker errors with MSVC,
   * so no calling of the empty constructor here.*/ {
    processArg(arg);
    // Expand parameter pack in C++ 11 way:
    int dummy[] = { 0, ( (void) processArg(moreArgsOrName), 0) ... };
    (void)dummy;
  };

  /// Construct from iterators.
  /// \tparam Iterator_t An iterator pointing to RooFit objects or references thereof.
  /// \param beginIt Iterator to first element to add.
  /// \param end Iterator to end of range to be added.
  /// \param name Optional name of the collection.
  template<typename Iterator_t,
      typename value_type = typename std::iterator_traits<Iterator_t>::value_type,
      typename = std::enable_if<std::is_convertible<const value_type*, const RooAbsArg*>::value> >
  RooArgSet(Iterator_t beginIt, Iterator_t endIt, const char* name="") :
  RooArgSet(name) {
    for (auto it = beginIt; it != endIt; ++it) {
      add(*it);
    }
  }

  ~RooArgSet() override;
  RooArgSet(const RooArgSet& other, const char *name="");
  TObject* clone(const char* newname) const override { return new RooArgSet(*this,newname); }
  TObject* create(const char* newname) const override { return new RooArgSet(newname); }
  RooArgSet& operator=(const RooArgSet& other) { RooAbsCollection::operator=(other) ; return *this ;}

  using RooAbsCollection::add;
  Bool_t add(const RooAbsArg& var, Bool_t silent=kFALSE) override;

  using RooAbsCollection::addOwned;
  Bool_t addOwned(RooAbsArg& var, Bool_t silent=kFALSE) override;

  using RooAbsCollection::addClone;
  RooAbsArg *addClone(const RooAbsArg& var, Bool_t silent=kFALSE) override;

  using RooAbsCollection::operator[];
  RooAbsArg& operator[](const TString& str) const;


  /// Shortcut for readFromStream(std::istream&, Bool_t, const char*, const char*, Bool_t), setting
  /// `flagReadAtt` and `section` to 0.
  virtual Bool_t readFromStream(std::istream& is, Bool_t compact, Bool_t verbose=kFALSE) {
    // I/O streaming interface (machine readable)
    return readFromStream(is, compact, 0, 0, verbose) ;
  }
  Bool_t readFromStream(std::istream& is, Bool_t compact, const char* flagReadAtt, const char* section, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(std::ostream& os, Bool_t compact, const char* section=0) const;  
  void writeToFile(const char* fileName) const ;
  Bool_t readFromFile(const char* fileName, const char* flagReadAtt=0, const char* section=0, Bool_t verbose=kFALSE) ;

  // Utilities functions when used as configuration object
  Double_t getRealValue(const char* name, Double_t defVal=0, Bool_t verbose=kFALSE) const ;
  const char* getCatLabel(const char* name, const char* defVal="", Bool_t verbose=kFALSE) const ;
  Int_t getCatIndex(const char* name, Int_t defVal=0, Bool_t verbose=kFALSE) const ;
  const char* getStringValue(const char* name, const char* defVal="", Bool_t verbose=kFALSE) const ;
  Bool_t setRealValue(const char* name, Double_t newVal=0, Bool_t verbose=kFALSE) ;
  Bool_t setCatLabel(const char* name, const char* newVal="", Bool_t verbose=kFALSE) ;
  Bool_t setCatIndex(const char* name, Int_t newVal=0, Bool_t verbose=kFALSE) ;
  Bool_t setStringValue(const char* name, const char* newVal="", Bool_t verbose=kFALSE) ;

  static void cleanup() ;

  Bool_t isInRange(const char* rangeSpec) ;

  /// Use RooAbsCollection::snapshot(), but return as RooArgSet.
  RooArgSet * snapshot(bool deepCopy = true) const {
    return static_cast<RooArgSet*>(RooAbsCollection::snapshot(deepCopy));
  }

  /// \copydoc RooAbsCollection::snapshot()
  Bool_t snapshot(RooAbsCollection& output, Bool_t deepCopy=kTRUE) const {
    return RooAbsCollection::snapshot(output, deepCopy);
  }

protected:
  Bool_t checkForDup(const RooAbsArg& arg, Bool_t silent) const ;

private:
  void processArg(const RooAbsArg& var) { add(var); }
  void processArg(const RooArgSet& set) { add(set); if (_name.Length() == 0) _name = set.GetName(); }
  void processArg(const RooArgList& list);
  void processArg(const char* name) { _name = name; }

#ifdef USEMEMPOOLFORARGSET
private:
  typedef MemPoolForRooSets<RooArgSet, 10*600> MemPool; //600 = about 100 kb
  //Initialise a static mem pool. It has to happen inside a function to solve the
  //static initialisation order fiasco. At the end of the program, this might have
  //to leak depending if RooArgSets are still alive. This depends on the order of destructions.
  static MemPool* memPool();
#endif
  
  ClassDefOverride(RooArgSet,1) // Set of RooAbsArg objects
};

#endif
