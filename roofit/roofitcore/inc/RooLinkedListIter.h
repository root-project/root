/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooLinkedListIter.h,v 1.11 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_LINKED_LIST_ITER
#define ROO_LINKED_LIST_ITER

//#include "Rtypes.h"
#include "TIterator.h"
#include "RooLinkedList.h"
#include <memory>
#include <assert.h>


///Interface for RooFIter-compatible iterators
class GenericRooFIter
{
  public:
  virtual RooAbsArg * next() = 0;
  virtual ~GenericRooFIter() {}
};

///A one-time forward iterator working on RooLinkedList or RooAbsCollection
class RooFIter
{
  public:
  RooFIter(std::shared_ptr<GenericRooFIter> && itImpl) : fIterImpl{std::move(itImpl)} {}
  RooFIter(const RooFIter &) = delete;
  RooFIter(RooFIter &&) = default;
  RooFIter & operator=(const RooFIter &) = delete;
  RooFIter & operator=(RooFIter &&) = default;

  RooAbsArg *next() {
    return fIterImpl->next();
  }

  private:
  std::shared_ptr<GenericRooFIter> fIterImpl;
};


class RooFIterForLinkedList final : public GenericRooFIter
{
  public:
  RooFIterForLinkedList() {}
  RooFIterForLinkedList(const RooLinkedList* list) : fPtr (list->_first) {}

  /// Return next element in collection
  RooAbsArg *next() override {
    if (!fPtr) return nullptr ;
    TObject* arg = fPtr->_arg ;
    fPtr = fPtr->_next;
    return (RooAbsArg*) arg ;
  }
    
 private:
    const RooLinkedListElem * fPtr{nullptr};  //! Next link element
};



template<class STLContainer>
class TIteratorToSTLInterface final : public TIterator , public GenericRooFIter {
  public:

  TIteratorToSTLInterface(const STLContainer & container) :
    fSTLContainer{&container},
    fSTLIter{fSTLContainer->begin()}
#ifndef NDEBUG
    ,fCurrentElem{fSTLContainer->empty() ? nullptr : *fSTLIter}
#endif
  {

  }

  TIterator & operator=(const TIterator &) override {
    throw;
  }

  const TCollection *GetCollection() const override {
    return nullptr;
  }

  RooAbsArg * next() override {
    if (atEnd())
      return nullptr;

#ifndef NDEBUG
    return nextChecked();
#endif

    return *fSTLIter++;
  }

  TObject * Next() override {
    return static_cast<TObject*>(next());
  }

  void Reset() override {
    if (fSTLContainer) {
      fSTLIter = fSTLContainer->begin();
#ifndef NDEBUG
      fCurrentElem = fSTLContainer->empty() ? nullptr : *fSTLIter;
#endif
    }
  }

  Bool_t operator!=(const TIterator & other) const override {
    if (!fSTLContainer)
      return true;

    const auto * castedOther =
        dynamic_cast<const TIteratorToSTLInterface<STLContainer>*>(&other);
    return !castedOther || fSTLContainer != castedOther->fSTLContainer
        || fSTLIter != castedOther->fSTLIter;
  }

  TObject * operator*() const override {
    if (atEnd())
      return nullptr;

#ifndef NDEBUG
    assert(fCurrentElem == *fSTLIter);
#endif

    return static_cast<TObject*>(*fSTLIter);
  }
  
  void invalidate() {
    fIsValid = false;
  }

  private:
  bool atEnd() const {
    if (!fIsValid) {
      throw std::logic_error("This iterator was invalidated by modifying the underlying collection.");
    }
    return !fSTLContainer
        || fSTLContainer->empty()
        || fSTLIter == fSTLContainer->end();
  }

  RooAbsArg * nextChecked();

  const STLContainer * fSTLContainer{nullptr}; //!
  typename STLContainer::const_iterator fSTLIter; //!
  bool fIsValid{true}; //!
#ifndef NDEBUG
  const RooAbsArg * fCurrentElem; //!
#endif
};


////////////////////////////////////////////////////////////////////////////////////////////
/// A wrapper around TIterator derivatives.
///
/// Its sole purpose is to act on the outside like the old RooLinkedListIter, even though
/// the underlying implementation may work an a totally different container, like e.g.
/// an STL container. This is needed to not break user code that is using a RooLinkedList or
/// a RooAbsCollection.
///
/// The iterator forwards all calls to the underlying iterator implementation.
/// All code using this iterator as an iterator over a RooAbsCollection should try to move away
/// from it because begin() and end() in RooAbsCollection are faster.
class RooLinkedListIter final : public TIterator {

  public:
  RooLinkedListIter(std::shared_ptr<TIterator> iterImpl) :
    fIterImpl{std::move(iterImpl)} {

  }

  RooLinkedListIter(const RooLinkedListIter &) = delete;
  RooLinkedListIter(RooLinkedListIter &&) = default;
  RooLinkedListIter & operator=(const RooLinkedListIter &) = delete;
  RooLinkedListIter & operator=(RooLinkedListIter &&) = default;

  TIterator &operator=(const TIterator & other) override {fIterImpl->operator=(other); return *this;}
  const TCollection *GetCollection() const override {return nullptr;}

  TObject * Next() override {return fIterImpl->Next();}
  void Reset() override {fIterImpl->Reset();}
  Bool_t operator!=(const TIterator & other) const override {return fIterImpl->operator!=(other);}
  TObject * operator*() const override {return fIterImpl->operator*();}

  private:
  std::shared_ptr<TIterator> fIterImpl; //!
};


class RooLinkedListIterImpl final : public TIterator {
public:

  RooLinkedListIterImpl() {
    // coverity[UNINIT_CTOR]
  } ;


  RooLinkedListIterImpl(const RooLinkedList* list, Bool_t forward) :
    TIterator(), _list(list), _ptr(forward ? _list->_first : _list->_last),
      _forward(forward)
  { }

  RooLinkedListIterImpl(const RooLinkedListIterImpl& other) :
    TIterator(other), _list(other._list), _ptr(other._ptr),
    _forward(other._forward)
  {
    // Copy constructor
  }
  
  virtual ~RooLinkedListIterImpl() { }
  
  TIterator& operator=(const TIterator& other) {

    // Iterator assignment operator

    if (&other==this) return *this ;
    const RooLinkedListIterImpl* iter = dynamic_cast<const RooLinkedListIterImpl*>(&other) ;
    if (iter) {
      _list = iter->_list ;
      _ptr = iter->_ptr ;
      _forward = iter->_forward ;
    }
    return *this ;
  }
    
  virtual const TCollection *GetCollection() const { 
    // Dummy
    return 0 ; 
  }

  virtual TObject *Next() { 
    // Return next element in collection
    if (!_ptr) return 0 ;
    TObject* arg = _ptr->_arg ;      
    _ptr = _forward ? _ptr->_next : _ptr->_prev ;
    return arg ;
  }

  TObject *NextNV() { 
    // Return next element in collection
    if (!_ptr) return 0 ;
    TObject* arg = _ptr->_arg ;      
    _ptr = _forward ? _ptr->_next : _ptr->_prev ;
    return arg ;
  }
  

  virtual void Reset() { 
    // Return iterator to first element in collection
    _ptr = _forward ? _list->_first : _list->_last ;
  }

  bool operator!=(const TIterator &aIter) const {
    const RooLinkedListIterImpl *iter(dynamic_cast<const RooLinkedListIterImpl*>(&aIter));
    if (iter) return (_ptr != iter->_ptr);
    return false; // for base class we don't implement a comparison
  }

  bool operator!=(const RooLinkedListIterImpl &aIter) const {
    return (_ptr != aIter._ptr);
  }

  virtual TObject *operator*() const {
    // Return element iterator points to
    return (_ptr ? _ptr->_arg : nullptr);
  }

protected:
  const RooLinkedList* _list ;     //! Collection iterated over
  const RooLinkedListElem* _ptr ;  //! Next link element
  Bool_t _forward ;                //  Iterator direction

//  ClassDef(RooLinkedListIterImpl,1) // Iterator for RooLinkedList container class
} ;




#endif
