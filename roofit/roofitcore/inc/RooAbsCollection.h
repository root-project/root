/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsCollection.h,v 1.26 2007/08/09 19:55:47 wouter Exp $
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
#ifndef ROO_ABS_COLLECTION
#define ROO_ABS_COLLECTION

#include "TObject.h"
#include "TString.h"
#include "RooAbsArg.h"
#include "RooPrintable.h"
#include "RooCmdArg.h"
#include "RooLinkedListIter.h"

// The range casts are not used in this file, but if you want to work with
// RooFit collections you also want to have static_range_cast and
// dynamic_range_cast available without including RangeCast.h every time.
#include "ROOT/RRangeCast.hxx"

#include "ROOT/RSpan.hxx"

#include <string>
#include <unordered_map>
#include <vector>
#include <type_traits>
#include <memory>


// To make ROOT::RangeStaticCast available under the name static_range_cast.
template <typename T, typename Range_t>
ROOT::RRangeCast<T, false, Range_t> static_range_cast(Range_t &&coll)
{
   return ROOT::RangeStaticCast<T>(std::forward<Range_t>(coll));
}


// To make ROOT::RangeDynCast available under the dynamic_range_cast.
template <typename T, typename Range_t>
ROOT::RRangeCast<T, true, Range_t> dynamic_range_cast(Range_t &&coll)
{
   return ROOT::RangeDynCast<T>(std::forward<Range_t>(coll));
}


class RooCmdArg;

namespace RooFit {
namespace Detail {
struct HashAssistedFind;
}
}

class RooAbsCollection : public TObject, public RooPrintable {
public:
  using Storage_t = std::vector<RooAbsArg*>;
  using const_iterator = Storage_t::const_iterator;


  // Constructors, assignment etc.
  RooAbsCollection();
  RooAbsCollection(const char *name);
  virtual TObject* clone(const char* newname) const = 0 ;
  virtual TObject* create(const char* newname) const = 0 ;
  virtual TObject* Clone(const char* newname=0) const { 
    return clone(newname?newname:GetName()) ; 
  }
  virtual ~RooAbsCollection();

  // Create a copy of an existing list. New variables cannot be added
  // to a copied list. The variables in the copied list are independent
  // of the original variables.
  RooAbsCollection(const RooAbsCollection& other, const char *name="");
  RooAbsCollection& operator=(const RooAbsCollection& other);

  void assign(const RooAbsCollection& other) const;
  RooAbsCollection &assignValueOnly(const RooAbsCollection& other, bool forceIfSizeOne=false);
  void assignFast(const RooAbsCollection& other, bool setValDirty=true) const;

  // Move constructor
  RooAbsCollection(RooAbsCollection && other);

  // Copy list and contents (and optionally 'deep' servers)
  RooAbsCollection *snapshot(Bool_t deepCopy=kTRUE) const ;
  Bool_t snapshot(RooAbsCollection& output, Bool_t deepCopy=kTRUE) const ;

  /// Set the size at which the collection will automatically start using an extra
  /// lookup table instead of performing a linear search.
  void setHashTableSize(Int_t number) {
    _sizeThresholdForMapSearch = number;
  }
  /// Query the size at which the collection will automatically start using an extra
  /// lookup table instead of performing a linear search.
  Int_t getHashTableSize() const { 
    return _sizeThresholdForMapSearch;
  }

  /// Const access to the underlying stl container.
  Storage_t const& get() const { return _list; }

  // List content management
  virtual Bool_t add(const RooAbsArg& var, Bool_t silent=kFALSE) ;
  virtual Bool_t addOwned(RooAbsArg& var, Bool_t silent=kFALSE);
  bool addOwned(std::unique_ptr<RooAbsArg> var, bool silent=false);
  virtual RooAbsArg *addClone(const RooAbsArg& var, Bool_t silent=kFALSE) ;
  virtual Bool_t replace(const RooAbsArg& var1, const RooAbsArg& var2) ;
  virtual Bool_t remove(const RooAbsArg& var, Bool_t silent=kFALSE, Bool_t matchByNameOnly=kFALSE) ;
  virtual void removeAll() ;

  template<typename Iterator_t,
      typename value_type = typename std::remove_pointer<typename std::iterator_traits<Iterator_t>::value_type>,
      typename = std::enable_if<std::is_convertible<const value_type*, const RooAbsArg*>::value> >
  bool add(Iterator_t beginIt, Iterator_t endIt, bool silent=false) {
    bool result = false ;
    _list.reserve(_list.size() + std::distance(beginIt, endIt));
    for (auto it = beginIt; it != endIt; ++it) {
      result |= add(**it,silent);
    }
    return result;
  }
  ////////////////////////////////////////////////////////////////////////////////
  /// Add a collection of arguments to this collection by calling add()
  /// for each element in the source collection
  bool add(const RooAbsCollection& list, bool silent=kFALSE) {
    return add(list._list.begin(), list._list.end(), silent);
  }
  virtual bool addOwned(const RooAbsCollection& list, bool silent=false);
  bool addOwned(RooAbsCollection&& list, bool silent=false);
  virtual void   addClone(const RooAbsCollection& list, Bool_t silent=kFALSE);
  Bool_t replace(const RooAbsCollection &other);
  Bool_t remove(const RooAbsCollection& list, Bool_t silent=kFALSE, Bool_t matchByNameOnly=kFALSE) ;
  template<class forwardIt>
  void remove(forwardIt rangeBegin, forwardIt rangeEnd, Bool_t silent = kFALSE, Bool_t matchByNameOnly = kFALSE) {
      for (forwardIt it = rangeBegin; it != rangeEnd; ++it) {
        static_assert(std::is_same<
            typename std::iterator_traits<forwardIt>::value_type,
            RooAbsArg*>::value, "Can only remove lists of RooAbsArg*.");
        auto castedElm = static_cast<RooAbsArg*>(*it);
        remove(*castedElm, silent, matchByNameOnly);
      }
  }

   // Utilities functions when used as configuration object
   Double_t getRealValue(const char* name, Double_t defVal=0, Bool_t verbose=kFALSE) const ;
   const char* getCatLabel(const char* name, const char* defVal="", Bool_t verbose=kFALSE) const ;
   Int_t getCatIndex(const char* name, Int_t defVal=0, Bool_t verbose=kFALSE) const ;
   const char* getStringValue(const char* name, const char* defVal="", Bool_t verbose=kFALSE) const ;
   Bool_t setRealValue(const char* name, Double_t newVal=0, Bool_t verbose=kFALSE) ;
   Bool_t setCatLabel(const char* name, const char* newVal="", Bool_t verbose=kFALSE) ;
   Bool_t setCatIndex(const char* name, Int_t newVal=0, Bool_t verbose=kFALSE) ;
   Bool_t setStringValue(const char* name, const char* newVal="", Bool_t verbose=kFALSE) ;

  // Group operations on AbsArgs
  void setAttribAll(const Text_t* name, Bool_t value=kTRUE) ;

  // List search methods
  RooAbsArg *find(const char *name) const ;
  RooAbsArg *find(const RooAbsArg&) const ;

  /// Find object by name in the collection
  TObject* FindObject(const char* name) const { return find(name); }

  /// Find object in the collection, Note: matching by object name, like the find() method
  TObject* FindObject(const TObject* obj) const { auto arg = dynamic_cast<const RooAbsArg*>(obj); return (arg) ? find(*arg) : nullptr; }

  /// Check if collection contains an argument with the same name as var.
  /// To check for a specific instance, use containsInstance().
  Bool_t contains(const RooAbsArg& var) const { 
    return find(var) != nullptr;
  }
  /// Check if this exact instance is in this collection.
  virtual Bool_t containsInstance(const RooAbsArg& var) const { 
    return std::find(_list.begin(), _list.end(), &var) != _list.end();
  }
  RooAbsCollection* selectByAttrib(const char* name, Bool_t value) const ;
  bool selectCommon(const RooAbsCollection& refColl, RooAbsCollection& outColl) const ;
  RooAbsCollection* selectCommon(const RooAbsCollection& refColl) const ;
  RooAbsCollection* selectByName(const char* nameList, Bool_t verbose=kFALSE) const ;
  Bool_t equals(const RooAbsCollection& otherColl) const ; 
  bool hasSameLayout(const RooAbsCollection& other) const;

  template<typename Iterator_t,
      typename value_type = typename std::remove_pointer<typename std::iterator_traits<Iterator_t>::value_type>,
      typename = std::enable_if<std::is_convertible<const value_type*, const RooAbsArg*>::value> >
  bool overlaps(Iterator_t otherCollBegin, Iterator_t otherCollEnd) const  {
    for (auto it = otherCollBegin; it != otherCollEnd; ++it) {
      if (find(**it)) {
        return true ;
      }
    }
    return false ;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Check if this and other collection have common entries
  bool overlaps(const RooAbsCollection& otherColl) const {
    return overlaps(otherColl._list.begin(), otherColl._list.end());
  }

  /// TIterator-style iteration over contained elements.
  /// \note These iterators are slow. Use begin() and end() or
  /// range-based for loop instead.
  inline TIterator* createIterator(Bool_t dir = kIterForward) const
  R__SUGGEST_ALTERNATIVE("begin(), end() and range-based for loops.") {
    // Create and return an iterator over the elements in this collection
    return new RooLinkedListIter(makeLegacyIterator(dir));
  }

  /// TIterator-style iteration over contained elements.
  /// \note This iterator is slow. Use begin() and end() or range-based for loop instead.
  RooLinkedListIter iterator(Bool_t dir = kIterForward) const
  R__SUGGEST_ALTERNATIVE("begin(), end() and range-based for loops.") {
    return RooLinkedListIter(makeLegacyIterator(dir));
  }

  /// One-time forward iterator.
  /// \note Use begin() and end() or range-based for loop instead.
  RooFIter fwdIterator() const
  R__SUGGEST_ALTERNATIVE("begin(), end() and range-based for loops.") {
    return RooFIter(makeLegacyIterator());
  }

  const_iterator begin() const {
    return _list.begin();
  }
  
  const_iterator end() const {
    return _list.end();
  }

  Storage_t::const_reverse_iterator rbegin() const {
    return _list.rbegin();
  }

  Storage_t::const_reverse_iterator rend() const {
      return _list.rend();
    }

  Storage_t::size_type size() const {
    return _list.size();
  }

  bool empty() const {
    return _list.empty();
  }

  void reserve(Storage_t::size_type count) {
    _list.reserve(count);
  }

  /// Clear contents. If the collection is owning, it will also delete the contents.
  void clear() {
    removeAll();
  }

  inline Int_t getSize() const { 
    // Return the number of elements in the collection
    return _list.size();
  }
  
  inline RooAbsArg *first() const { 
    // Return the first element in this collection
    return _list.front();
  }

  RooAbsArg * operator[](Storage_t::size_type i) const {
    return _list[i];
  }


  /// Returns index of given arg, or -1 if arg is not in the collection.
  inline Int_t index(const RooAbsArg* arg) const {
    auto item = std::find(_list.begin(), _list.end(), arg);
    return item != _list.end() ? item - _list.begin() : -1;
  }

  /// Returns index of given arg, or -1 if arg is not in the collection.
  inline Int_t index(const RooAbsArg& arg) const {
    return index(&arg);
  }

  Int_t index(const char* name) const;

  inline virtual void Print(Option_t *options= 0) const {
    // Printing interface (human readable)
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }
  std::string contentsString() const ;


  virtual void printName(std::ostream& os) const ;
  virtual void printTitle(std::ostream& os) const ;
  virtual void printClassName(std::ostream& os) const ;
  virtual void printValue(std::ostream& os) const ;
  virtual void printMultiline(std::ostream& os, Int_t contents, Bool_t verbose=kFALSE, TString indent="") const ;

  virtual Int_t defaultPrintContents(Option_t* opt) const ;

  // Latex printing methods
  void printLatex(const RooCmdArg& arg1=RooCmdArg(), const RooCmdArg& arg2=RooCmdArg(),	
		  const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),	
		  const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg(),	
		  const RooCmdArg& arg7=RooCmdArg(), const RooCmdArg& arg8=RooCmdArg()) const ;
  void printLatex(std::ostream& ofs, Int_t ncol, const char* option="NEYU", Int_t sigDigit=1, 
                  const RooLinkedList& siblingLists=RooLinkedList(), const RooCmdArg* formatCmd=0) const ;

  void setName(const char *name) {
    // Set name of collection
    _name= name; 
  }
  const char* GetName() const { 
    // Return namer of collection
    return _name.Data() ; 
  }
  Bool_t isOwning() const { 
    // Does collection own contents?
    return _ownCont ; 
  }

  Bool_t allInRange(const char* rangeSpec) const ;

  void dump() const ;

  void releaseOwnership() { _ownCont = kFALSE ; }
  void takeOwnership() { _ownCont = kTRUE ; }

  void sort(Bool_t reverse = false);
  void sortTopologically();

  virtual void RecursiveRemove(TObject *obj);

  void useHashMapForFind(bool flag) const;

  // For use in the RooArgList/Set(std::vector<RooAbsArgPtrOrDouble> const&) constructor.
  // Can be replaced with std::variant when C++17 is the minimum supported standard.
  struct RooAbsArgPtrOrDouble {
    RooAbsArgPtrOrDouble(RooAbsArg & arg) : ptr{&arg}, hasPtr{true} {}
    RooAbsArgPtrOrDouble(double x) : val{x}, hasPtr{false} {}

    RooAbsArg * ptr = nullptr;
    double val = 0.0;
    bool hasPtr = false;
  };

protected:
  Storage_t _list; // Actual object storage
  using LegacyIterator_t = TIteratorToSTLInterface<Storage_t>;

  Bool_t _ownCont;  // Flag to identify a list that owns its contents.
  TString _name;    // Our name.
  Bool_t _allRRV ; // All contents are RRV

  void deleteList() ;

  // Support for snapshot method 
  Bool_t addServerClonesToList(const RooAbsArg& var) ;

  inline TNamed* structureTag() { if (_structureTag==0) makeStructureTag() ; return _structureTag ; }
  inline TNamed* typedStructureTag() { if (_typedStructureTag==0) makeTypedStructureTag() ; return _typedStructureTag ; }

  mutable TNamed* _structureTag{nullptr}; //! Structure tag
  mutable TNamed* _typedStructureTag{nullptr}; //! Typed structure tag
  
  inline void clearStructureTags() { _structureTag=0 ; _typedStructureTag = 0 ; }

  void makeStructureTag() ;
  void makeTypedStructureTag() ;

  /// Determine whether it's possible to add a given RooAbsArg to the collection or not.
  virtual bool canBeAdded(const RooAbsArg& arg, bool silent) const = 0;

  template<class T>
  static void assert_is_no_temporary(T &&) {
    static_assert(!std::is_rvalue_reference<T&&>::value,
      "A reference to a temporary RooAbsArg will be passed to a RooAbsCollection constructor! "
      "This is not allowed, because the collection will not own the arguments. "
      "Hence, the collection will contain dangling pointers when the temporary goes out of scope."
    );
  }

private:
  std::unique_ptr<LegacyIterator_t> makeLegacyIterator (bool forward = true) const;

  using HashAssistedFind = RooFit::Detail::HashAssistedFind;
  mutable std::unique_ptr<HashAssistedFind> _hashAssistedFind; //!
  std::size_t _sizeThresholdForMapSearch; //!

  void insert(RooAbsArg*);

  ClassDef(RooAbsCollection,3) // Collection of RooAbsArg objects
};

#endif
