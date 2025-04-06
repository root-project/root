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

#include <RooAbsArg.h>
#include <RooCmdArg.h>
#include <RooFit/UniqueId.h>
#include <RooLinkedListIter.h>
#include <RooPrintable.h>

// The range casts are not used in this file, but if you want to work with
// RooFit collections you also want to have static_range_cast and
// dynamic_range_cast available without including RangeCast.h every time.
#include <ROOT/RRangeCast.hxx>

#include <ROOT/RSpan.hxx>

#include <TClass.h>
#include <TObject.h>
#include <TString.h>

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
  virtual TObject* clone(const char* newname=nullptr) const = 0 ;
  virtual TObject* create(const char* newname) const = 0 ;
  TObject* Clone(const char* newname=nullptr) const override {
    return clone(newname?newname:GetName()) ;
  }
  ~RooAbsCollection() override;

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

  /// Returns a unique ID that is different for every instantiated
  /// RooAbsCollection. This ID can be used to check whether two collections
  /// are the same object, which is safer than memory address comparisons that
  /// might result in false positives when memory is recycled.
  RooFit::UniqueId<RooAbsCollection> const& uniqueId() const { return _uniqueId; }

  // Copy list and contents (and optionally 'deep' servers)
  RooAbsCollection *snapshot(bool deepCopy=true) const ;
  bool snapshot(RooAbsCollection& output, bool deepCopy=true) const ;

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
  virtual bool add(const RooAbsArg& var, bool silent=false) ;
// The following function is not memory safe, because it takes ownership of var
// without moving it. It is not publicly available in the memory safe
// interfaces mode.
#ifdef ROOFIT_MEMORY_SAFE_INTERFACES
protected:
#endif
  virtual bool addOwned(RooAbsArg& var, bool silent=false);
#ifdef ROOFIT_MEMORY_SAFE_INTERFACES
public:
#endif
  bool addOwned(std::unique_ptr<RooAbsArg> var, bool silent=false);
  virtual RooAbsArg *addClone(const RooAbsArg& var, bool silent=false) ;
  virtual bool replace(const RooAbsArg& var1, const RooAbsArg& var2) ;
  bool replace(RooAbsArg* var1, std::unique_ptr<RooAbsArg> var2) ;
  virtual bool remove(const RooAbsArg& var, bool silent=false, bool matchByNameOnly=false) ;
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
  bool add(const RooAbsCollection& list, bool silent=false) {
    return add(list._list.begin(), list._list.end(), silent);
  }
   /**
    * \brief Adds elements of a given RooAbsCollection to the container if they match the specified type.
    *
    * This function iterates through the elements of the provided RooAbsCollection and checks if each element
    * matches the specified type. If any element doesn't match the type, it throws an exception.
    *
    * \tparam Arg_t The type to match for elements in the collection. Must inherit from RooAbsArg.
    * \param list The RooAbsCollection containing elements to be added.
    * \param silent Forwarded to the non-typed add function. If true,
    *               suppresses error messages when adding elements, e.g. when
    *               the collection is a RooArgSet and the element is already in
    *               the set.
    * \return Returns true if all elements could be added, else false.
    *
    * \throws std::invalid_argument if an element in the collection doesn't match the specified type.
    */
   template <class Arg_t>
   bool addTyped(const RooAbsCollection &list, bool silent = false)
   {
      for (RooAbsArg *arg : list) {
         if (!dynamic_cast<Arg_t *>(arg)) {
            throwAddTypedException(Arg_t::Class(), arg);
         }
      }
      return add(list, silent);
   }
  virtual bool addOwned(const RooAbsCollection& list, bool silent=false);
  bool addOwned(RooAbsCollection&& list, bool silent=false);
  virtual void   addClone(const RooAbsCollection& list, bool silent=false);
  bool replace(const RooAbsCollection &other);
  bool remove(const RooAbsCollection& list, bool silent=false, bool matchByNameOnly=false) ;
  template<class forwardIt>
  void remove(forwardIt rangeBegin, forwardIt rangeEnd, bool silent = false, bool matchByNameOnly = false) {
      for (forwardIt it = rangeBegin; it != rangeEnd; ++it) {
        static_assert(std::is_same<
            typename std::iterator_traits<forwardIt>::value_type,
            RooAbsArg*>::value, "Can only remove lists of RooAbsArg*.");
        auto castedElm = static_cast<RooAbsArg*>(*it);
        remove(*castedElm, silent, matchByNameOnly);
      }
  }

   // Utilities functions when used as configuration object
   double getRealValue(const char* name, double defVal=0.0, bool verbose=false) const ;
   const char* getCatLabel(const char* name, const char* defVal="", bool verbose=false) const ;
   Int_t getCatIndex(const char* name, Int_t defVal=0, bool verbose=false) const ;
   const char* getStringValue(const char* name, const char* defVal="", bool verbose=false) const ;
   bool setRealValue(const char* name, double newVal=0.0, bool verbose=false) ;
   bool setCatLabel(const char* name, const char* newVal="", bool verbose=false) ;
   bool setCatIndex(const char* name, Int_t newVal=0, bool verbose=false) ;
   bool setStringValue(const char* name, const char* newVal="", bool verbose=false) ;

  // Group operations on AbsArgs
  void setAttribAll(const Text_t* name, bool value=true) ;

  // List search methods
  RooAbsArg *find(const char *name) const ;
  RooAbsArg *find(const RooAbsArg&) const ;

  /// Find object by name in the collection
  TObject* FindObject(const char* name) const override { return find(name); }

  /// Find object in the collection, Note: matching by object name, like the find() method
  TObject* FindObject(const TObject* obj) const override { auto arg = dynamic_cast<const RooAbsArg*>(obj); return (arg) ? find(*arg) : nullptr; }

  /// Check if collection contains an argument with the same name as var.
  /// To check for a specific instance, use containsInstance().
  bool contains(const RooAbsArg& var) const {
    return find(var) != nullptr;
  }
  /// Check if this exact instance is in this collection.
  virtual bool containsInstance(const RooAbsArg& var) const {
    return std::find(_list.begin(), _list.end(), &var) != _list.end();
  }
  RooAbsCollection* selectByAttrib(const char* name, bool value) const ;
  bool selectCommon(const RooAbsCollection& refColl, RooAbsCollection& outColl) const ;
  RooAbsCollection* selectCommon(const RooAbsCollection& refColl) const ;
  RooAbsCollection* selectByName(const char* nameList, bool verbose=false) const ;
  bool equals(const RooAbsCollection& otherColl) const ;
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

  /// Return the number of elements in the collection
  inline Int_t getSize() const R__SUGGEST_ALTERNATIVE("size() returns true size.") {
    return _list.size();
  }

  inline RooAbsArg *first() const {
    // Return the first element in this collection
    // calling front on an empty container is undefined
    return _list.empty() ? nullptr : _list.front();
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

  inline void Print(Option_t *options= nullptr) const override {
    // Printing interface (human readable)
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }
  std::string contentsString() const ;


  void printName(std::ostream& os) const override ;
  void printTitle(std::ostream& os) const override ;
  void printClassName(std::ostream& os) const override ;
  void printValue(std::ostream& os) const override ;
  void printMultiline(std::ostream& os, Int_t contents, bool verbose=false, TString indent="") const override ;

  Int_t defaultPrintContents(Option_t* opt) const override ;

  // Latex printing methods
  void printLatex(const RooCmdArg& arg1={}, const RooCmdArg& arg2={},
        const RooCmdArg& arg3={}, const RooCmdArg& arg4={},
        const RooCmdArg& arg5={}, const RooCmdArg& arg6={},
        const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) const ;
  void printLatex(std::ostream& ofs, Int_t ncol, const char* option="NEYU", Int_t sigDigit=1,
                  const RooLinkedList& siblingLists=RooLinkedList(), const RooCmdArg* formatCmd=nullptr) const ;

  void setName(const char *name) {
    // Set name of collection
    _name= name;
  }
  const char* GetName() const override {
    // Return namer of collection
    return _name.Data() ;
  }
  bool isOwning() const {
    // Does collection own contents?
    return _ownCont ;
  }

  bool allInRange(const char* rangeSpec) const ;

  void dump() const ;

  void releaseOwnership() { _ownCont = false ; }
  void takeOwnership() { _ownCont = true ; }

  void sort(bool reverse = false);
  void sortTopologically();

  void RecursiveRemove(TObject *obj) override;

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
  Storage_t _list;  ///< Actual object storage
  using LegacyIterator_t = TIteratorToSTLInterface<Storage_t>;

  bool _ownCont = false; ///< Flag to identify a list that owns its contents.
  TString _name;           ///< Our name.
  bool _allRRV = true;   ///< All contents are RRV

  void deleteList() ;

  inline TNamed* structureTag() { if (_structureTag==nullptr) makeStructureTag() ; return _structureTag ; }
  inline TNamed* typedStructureTag() { if (_typedStructureTag==nullptr) makeTypedStructureTag() ; return _typedStructureTag ; }

  mutable TNamed* _structureTag{nullptr};      ///<! Structure tag
  mutable TNamed* _typedStructureTag{nullptr}; ///<! Typed structure tag

  inline void clearStructureTags() { _structureTag = nullptr ; _typedStructureTag = nullptr ; }

  void makeStructureTag() {}
  void makeTypedStructureTag() {}

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

  bool replaceImpl(const RooAbsArg& var1, const RooAbsArg& var2);

  using HashAssistedFind = RooFit::Detail::HashAssistedFind;
  mutable HashAssistedFind *_hashAssistedFind = nullptr; ///<!
  std::size_t _sizeThresholdForMapSearch = 100; ///<!

  void insert(RooAbsArg*);

  static void throwAddTypedException(TClass *klass, RooAbsArg *arg);

  const RooFit::UniqueId<RooAbsCollection> _uniqueId; //!

  ClassDefOverride(RooAbsCollection,3) // Collection of RooAbsArg objects
};

#endif
