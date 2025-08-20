// @(#)root/cont:$Id$
// Author: Philippe Canal 20/08/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualCollectionProxy
#define ROOT_TVirtualCollectionProxy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualCollectionProxy                                              //
//                                                                      //
// Virtual interface of a proxy object for a collection class           //
// In particular this is used to implement splitting, emulation,        //
// and TTreeFormula access to STL containers.                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClassRef.h"
#include "TDataType.h"

// Macro indicating the version of the Collection Proxy interface followed
// by this ROOT build (See also Reflex/Builder/CollectionProxy.h).
#define ROOT_COLLECTIONPROXY_VERSION 3

class TClass;
namespace TStreamerInfoActions {
   class TActionSequence;
}

// clang-format off
/**
\class TVirtualCollectionProxy
\brief Defines a common interface to inspect/change the contents of an object that represents a collection

Specifically, an object of a class that derives from TVirtualCollectionProxy relays accesses to any object that
matches the proxied collection type.
The interface provides two families of functions: (i) for direct manipulation, e.g. `Insert()` or `At()`; and
(ii) iterator-based, e.g. `GetFunctionCreateIterators()` or `GetFunctionNext()`.
TVirtualCollectionProxy objects are stateful; in particular, many functions require to set the object to operate
on via `PushProxy()` / `PopProxy()`.  The `TPushPop` RAII class is provided for convenience.
A collection proxy for a given class can be permanently set using `TClass::CopyCollectionProxy()`.
The `Generate()` function should be overridden in derived classes to return a clean object of the most-derived class.
*/
// clang-format on
class TVirtualCollectionProxy {
private:
   TVirtualCollectionProxy(const TVirtualCollectionProxy&) = delete;
   TVirtualCollectionProxy& operator=(const TVirtualCollectionProxy&) = delete;

protected:
   TClassRef fClass;
   UInt_t    fProperties;
   friend class TClass;

public:
   enum EProperty {
      // No longer used
      // kIsInitialized = BIT(1),
      kIsAssociative = BIT(2),
      kIsEmulated = BIT(3),
      /// The collection contains directly or indirectly (via other collection) some pointers that need explicit
      /// deletion
      kNeedDelete = BIT(4),
      kCustomAlloc = BIT(5) ///< The collection has a custom allocator.
   };

   /// RAII helper class that ensures that `PushProxy()` / `PopProxy()` are called when entering / leaving a C++ context
   class TPushPop {
   public:
      TVirtualCollectionProxy *fProxy;
      inline TPushPop(TVirtualCollectionProxy *proxy,
         void *objectstart) : fProxy(proxy) { fProxy->PushProxy(objectstart); }
      inline ~TPushPop() { fProxy->PopProxy(); }
   private:
      TPushPop(const TPushPop&) = delete;
      TPushPop& operator=(const TPushPop&) = delete;
   };

   TVirtualCollectionProxy() : fClass(), fProperties(0) {}
   TVirtualCollectionProxy(TClass *cl) : fClass(cl), fProperties(0) {}
   virtual ~TVirtualCollectionProxy() {}

   /// Returns a clean object of the actual class that derives from TVirtualCollectionProxy.  The caller is responsible
   /// for deleting the returned object.
   virtual TVirtualCollectionProxy *Generate() const = 0;

   /// Reset the information gathered from StreamerInfos and value's TClass.
   virtual Bool_t    Reset() { return kTRUE; }

   /// Return a pointer to the `TClass` representing the proxied _container_ class
   virtual TClass *GetCollectionClass() const { return fClass; }

   /// Return the type of the proxied collection (see enumeration TClassEdit::ESTLType)
   virtual Int_t GetCollectionType() const = 0;

   /// Return the offset between two consecutive in-memory values (which depends on the `sizeof()` and alignment of the
   /// value type).
   virtual ULong_t GetIncrement() const = 0;

   /// Return miscallenous properties of the proxy (see TVirtualCollectionProxy::EProperty)
   virtual Int_t GetProperties() const { return fProperties; }

   /// Construct a new container object and return its address
   virtual void *New() const { return !fClass.GetClass() ? nullptr : fClass->New(); }
   /// Construct a new container object at the address given by `arena`
   virtual void *New(void *arena) const { return !fClass.GetClass() ? nullptr : fClass->New(arena); }
   /// Construct a new container object and return its address
   virtual TClass::ObjectPtr NewObject() const
   {
      return !fClass.GetClass() ? TClass::ObjectPtr{} : fClass->NewObject();
   }
   /// Construct a new container object at the address given by `arena`
   virtual TClass::ObjectPtr NewObject(void *arena) const
   {
      return !fClass.GetClass() ? TClass::ObjectPtr{} : fClass->NewObject(arena);
   }

   /// Construct an array of `nElements` container objects and return the base address of the array
   virtual void *NewArray(Int_t nElements) const { return !fClass.GetClass() ? nullptr : fClass->NewArray(nElements); }
   /// Construct an array of `nElements` container objects at the address given by `arena`
   virtual void *NewArray(Int_t nElements, void *arena) const
   {
      return !fClass.GetClass() ? nullptr : fClass->NewArray(nElements, arena);
   }
   /// Construct an array of `nElements` container objects and return the base address of the array
   virtual TClass::ObjectPtr NewObjectArray(Int_t nElements) const
   {
      return !fClass.GetClass() ? TClass::ObjectPtr{} : fClass->NewObjectArray(nElements);
   }
   /// Construct an array of `nElements` container objects at the address given by `arena`
   virtual TClass::ObjectPtr NewObjectArray(Int_t nElements, void *arena) const
   {
      return !fClass.GetClass() ? TClass::ObjectPtr{} : fClass->NewObjectArray(nElements, arena);
   }

   /// Execute the container destructor
   virtual void Destructor(void *p, Bool_t dtorOnly = kFALSE) const
   {
      TClass* cl = fClass.GetClass();
      if (cl) cl->Destructor(p, dtorOnly);
   }

   /// Execute the container array destructor
   virtual void DeleteArray(void *p, Bool_t dtorOnly = kFALSE) const
   {
      TClass* cl = fClass.GetClass();
      if (cl) cl->DeleteArray(p, dtorOnly);
   }

   /// Return the `sizeof()` of the collection object
   virtual UInt_t Sizeof() const = 0;

   /// Set the address of the container being proxied and keep track of the previous one
   virtual void PushProxy(void *objectstart) = 0;

   /// Reset the address of the container being proxied to the previous container
   virtual void PopProxy() = 0;

   /// Return `true` if the content is of type 'pointer to'
   virtual Bool_t HasPointers() const = 0;

   /// If the value type is a user-defined class, return a pointer to the `TClass` representing the
   /// value type of the container.
   virtual TClass *GetValueClass() const = 0;

   /// If the value type is a fundamental data type, return its type (see enumeration EDataType).
   virtual EDataType GetType() const = 0;

   /// Return the address of the value at index `idx`
   virtual void *At(UInt_t idx) = 0;

   /// Clear the container
   virtual void Clear(const char *opt = "") = 0;

   /// Return the current number of elements in the container
   virtual UInt_t Size() const = 0;

   /// Allocates space for storing at least `n` elements.  This function returns a pointer to the actual object on
   /// which insertions should take place.  For associative collections, this function returns a pointer to a temporary
   /// buffer known as the staging area.  If the insertion happened in a staging area (i.e. the returned pointer !=
   /// proxied object), `Commit()` should be called on the value returned by this function.
   virtual void*     Allocate(UInt_t n, Bool_t forceDelete) = 0;

   /// Commits pending elements in a staging area (see Allocate() for more information).
   virtual void      Commit(void*) = 0;

   /// Insert elements into the proxied container.  `data` is a C-style array of the value type of the given `size`.
   /// For associative containers, e.g. `std::map`, the data type should be `std::pair<Key_t, Value_t>`.
   virtual void Insert(const void *data, void *container, size_t size) = 0;

   /// Return the address of the value at index `idx`
   char *operator[](UInt_t idx) const { return (char *)(const_cast<TVirtualCollectionProxy *>(this))->At(idx); }

   // Functions related to member-wise actions
   virtual TStreamerInfoActions::TActionSequence *GetConversionReadMemberWiseActions(TClass *oldClass, Int_t version) = 0;
   virtual TStreamerInfoActions::TActionSequence *GetReadMemberWiseActions(Int_t version) = 0;
   virtual TStreamerInfoActions::TActionSequence *GetWriteMemberWiseActions() = 0;

   /// The size of a small buffer that can be allocated on the stack to store iterator-specific information
   static const Int_t fgIteratorArenaSize = 16; // greater than sizeof(void*) + sizeof(UInt_t)

   /// `*begin_arena` and `*end_arena` should contain the location of a memory arena of size `fgIteratorArenaSize`.
   /// If iterator-specific information is of that size or less, the iterators will be constructed in place in the given
   /// locations.  Otherwise, iterators will be allocated via `new` and their address returned by modifying the value
   /// of `*begin_arena` and `*end_arena`.
   /// As a special case, given that iterators for array-backed containers are just pointers, the required information
   /// will be directly stored in `*(begin|end)_arena`.
   typedef void (*CreateIterators_t)(void *collection, void **begin_arena, void **end_arena, TVirtualCollectionProxy *proxy);

   /// Return a pointer to a function that can create an iterator pair, where each iterator points to the begin and end
   /// of the collection, respectively (see CreateIterators_t).  If `read == kTRUE`, data is to be read from disk, i.e.
   /// written to the in-memory collection.
   virtual CreateIterators_t GetFunctionCreateIterators(Bool_t read = kTRUE) = 0;

   /// Copy the iterator `source` into `dest`.  `dest` should contain the location of a memory arena of size
   /// `fgIteratorArenaSize`.
   /// If iterator-specific information is of that size or less, the iterators will be constructed in place in the given
   /// locations.  Otherwise, iterators will be allocated via `new` and their address returned by modifying the value
   /// of `*begin_arena` and `*end_arena`.  The actual address of the iterator is returned in any case.
   typedef void* (*CopyIterator_t)(void *dest, const void *source);

   /// Return a pointer to a function that can copy an iterator (see CopyIterator_t).  If `read == kTRUE`, data is to be
   /// read from disk, i.e. written to the in-memory collection.
   virtual CopyIterator_t GetFunctionCopyIterator(Bool_t read = kTRUE) = 0;

   /// `iter` and `end` should be pointers to an iterator to be incremented and an iterator that points to the end of
   /// the collection, respectively.  If `iter` has not reached the end of the collection, this function increments the
   /// iterator and returns a pointer to the element before the increment.  Otherwise, `nullptr` is returned.
   typedef void* (*Next_t)(void *iter, const void *end);

   /// Return a pointer to a function that can advance an iterator (see Next_t).  If `read == kTRUE`, data is to be
   /// read from disk, i.e. written to the in-memory collection.
   virtual Next_t GetFunctionNext(Bool_t read = kTRUE) = 0;

   /// If the size of the iterator is greater than `fgIteratorArenaSize`, call delete on the addresses; otherwise, just
   /// call the iterator's destructor.
   typedef void (*DeleteIterator_t)(void *iter);
   typedef void (*DeleteTwoIterators_t)(void *begin, void *end);

   /// Return a pointer to a function that can delete an iterator (pair) (see DeleteIterator_t).  If `read == kTRUE`,
   /// data is to be read from disk, i.e. written to the in-memory collection.
   virtual DeleteIterator_t GetFunctionDeleteIterator(Bool_t read = kTRUE) = 0;
   virtual DeleteTwoIterators_t GetFunctionDeleteTwoIterators(Bool_t read = kTRUE) = 0;
};

#endif
