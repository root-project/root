// @(#)root/cont:$Id$
// Author: Markus Frank  28/10/04. Philippe Canal 02/01/2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCollectionProxyInfo
#define ROOT_TCollectionProxyInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  Small helper to gather the information neede to generate a          //
//  Collection Proxy                                                    //
//
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TError
#include "TError.h"
#endif 
#include <vector>

#if defined(_WIN32)
   #if _MSC_VER<1300
      #define TYPENAME
      #define R__VCXX6
   #else
      #define TYPENAME typename
   #endif
#else
   #define TYPENAME typename
#endif

namespace ROOT {

   class TCollectionProxyInfo {
      // This class is a place holder for the information needed
      // to create the proper Collection Proxy.
      // This is similar to Reflex's CollFuncTable.

   public:

      // Same value as TVirtualCollectionProxy.
      static const UInt_t fgIteratorArenaSize = 16; // greater than sizeof(void*) + sizeof(UInt_t)

   /** @class template TCollectionProxyInfo::IteratorValue 
    *
    * Small helper to encapsulate whether to return the value
    * pointed to by the iterator or its address.
    *
    **/

      template <typename Cont_t, typename value> struct IteratorValue {
         static void* get(typename Cont_t::iterator &iter) {
            return (void*)&(*iter);
         }
      };

      template <typename Cont_t, typename value_ptr> struct IteratorValue<Cont_t, value_ptr*> {
         static void* get(typename Cont_t::iterator &iter) {
            return (void*)(*iter);
         }
      };

   /** @class template TCollectionProxyInfo::Iterators 
    *
    * Small helper to implement the function to create,access and destroy
    * iterators.
    *
    **/

      template <typename Cont_t, bool large = false> 
      struct Iterators {
         typedef Cont_t *PCont_t;
         typedef typename Cont_t::iterator iterator;

         static void create(void *coll, void **begin_arena, void **end_arena, TVirtualCollectionProxy*) {
            PCont_t c = PCont_t(coll);
            new (*begin_arena) iterator(c->begin());
            new (*end_arena) iterator(c->end());
         }
         static void* copy(void *dest_arena, const void *source_ptr) {
            iterator *source = (iterator *)(source_ptr);
            new (dest_arena) iterator(*source);
            return dest_arena; 
         }
         static void* next(void *iter_loc, const void *end_loc) {
            iterator *end = (iterator *)(end_loc);
            iterator *iter = (iterator *)(iter_loc);
            if (*iter != *end) {
               void *result = IteratorValue<Cont_t, typename Cont_t::value_type>::get(*iter);
               ++(*iter);
               return result;
            }
            return 0;
         }
         static void destruct1(void *iter_ptr) {
            iterator *start = (iterator *)(iter_ptr);
            start->~iterator();
         }
         static void destruct2(void *begin_ptr, void *end_ptr) {
            iterator *start = (iterator *)(begin_ptr);
            iterator *end = (iterator *)(end_ptr);
            start->~iterator();
            end->~iterator();
         }
      };

      // For Vector we take an extra short cut to avoid derefencing
      // the iterator all the time and redefine the 'address' of the
      // iterator as the iterator itself.  This requires special handling
      // in the looper (see TStreamerInfoAction) but is much faster.
      template <typename T> struct Iterators<std::vector<T>, false> {
         typedef std::vector<T> Cont_t;
         typedef Cont_t *PCont_t;
         typedef typename Cont_t::iterator iterator;

         static void create(void *coll, void **begin_arena, void **end_arena, TVirtualCollectionProxy*) {
            PCont_t c = PCont_t(coll);
            if (c->empty()) {
               *begin_arena = 0;
               *end_arena = 0;
               return;
            }
            *begin_arena = &(*c->begin());
#ifdef R__VISUAL_CPLUSPLUS
            *end_arena = &(*(c->end()-1)) + 1; // On windows we can not dererence the end iterator at all.
#else
            // coverity[past_the_end] Safe on other platforms
            *end_arena = &(*c->end());
#endif
         }
         static void* copy(void *dest, const void *source) {
            *(void**)dest = *(void**)(const_cast<void*>(source));
            return dest; 
         }
         static void* next(void * /* iter_loc */, const void * /* end_loc */) {
            // Should not be used.
            R__ASSERT(0 && "Intentionally not implemented, do not use.");
            return 0;
         }
         static void destruct1(void  * /* iter_ptr */) {
            // Nothing to do
         }
         static void destruct2(void * /* begin_ptr */, void * /* end_ptr */) {
            // Nothing to do
         }
      };

      template <typename Cont_t> struct Iterators<Cont_t, /* large= */ true > {
         typedef Cont_t *PCont_t;
         typedef typename Cont_t::iterator iterator;

         static void create(void *coll, void **begin_arena, void **end_arena, TVirtualCollectionProxy*) {
            PCont_t  c = PCont_t(coll);
            *begin_arena = new iterator(c->begin());
            *end_arena = new iterator(c->end());        
         }
         static void* copy(void * /*dest_arena*/, const void *source_ptr) {
            iterator *source = (iterator *)(source_ptr);
            void *iter = new iterator(*source);
            return iter;
         }
         static void* next(void *iter_loc, const void *end_loc) {
            iterator *end = (iterator *)(end_loc);
            iterator *iter = (iterator *)(iter_loc);
            if (*iter != *end) {
               void *result = IteratorValue<Cont_t, typename Cont_t::value_type>::get(*iter);
               ++(*iter);
               return result;
            }
            return 0;
         }
         static void destruct1(void *begin_ptr) {
            iterator *start = (iterator *)(begin_ptr);
            delete start;
         }
         static void destruct2(void *begin_ptr, void *end_ptr) {
            iterator *start = (iterator *)(begin_ptr);
            iterator *end = (iterator *)(end_ptr);
            delete start;
            delete end;
         }
      };

  /** @class TCollectionProxyInfo::Environ TCollectionProxyInfo.h TCollectionProxyInfo.h
    *
    * Small helper to save proxy environment in the event of
    * recursive calls.
    *
    * @author  M.Frank
    * @version 1.0
    * @date    10/10/2004
    */
#ifndef __CINT__
   struct EnvironBase {
   private:
      EnvironBase(const EnvironBase&); // Intentionally not implement, copy is not supported
      EnvironBase &operator=(const EnvironBase&); // Intentionally not implement, copy is not supported
   public:
      EnvironBase() : fIdx(0), fSize(0), fObject(0), fStart(0), fTemp(0), fUseTemp(kFALSE), fRefCount(1), fSpace(0)
      {
      }
      virtual ~EnvironBase() {}
      size_t              fIdx;
      size_t              fSize;
      void*               fObject;
      void*               fStart;
      void*               fTemp;
      union {
         Bool_t fUseTemp;
         Bool_t fLastValueVecBool;
      };
      int                 fRefCount;
      size_t              fSpace;      
   };
   template <typename T> struct Environ : public EnvironBase {
      typedef T           Iter_t;
      Iter_t              fIterator;
      T& iter() { return fIterator; }
      static void        *Create() {
         return new Environ();
      }
   };
#else
   struct EnvironBase;
   template <typename T> struct Environ;
#endif

   template <class T, class Q> struct PairHolder {
      T first;
      Q second;
      PairHolder() {}
      PairHolder(const PairHolder& c) : first(c.first), second(c.second) {}
      virtual ~PairHolder() {}
   private:
      PairHolder& operator=(const PairHolder&);  // not implemented
   };

   template <class T> struct Address {
      virtual ~Address() {}
      static void* address(T ref) {
         return (void*)&ref;
      }
   };

   /** @class TCollectionProxyInfo::Type TCollectionProxyInfo.h TCollectionProxyInfo.h
    *
    * Small helper to encapsulate basic data accesses for
    * all STL continers.
    *
    * @author  M.Frank
    * @version 1.0
    * @date    10/10/2004
    */
   template <class T> struct Type
      : public Address<TYPENAME T::const_reference>
   {
      typedef T                      Cont_t;
      typedef typename T::iterator   Iter_t;
      typedef typename T::value_type Value_t;
      typedef Environ<Iter_t>        Env_t;
      typedef Env_t                 *PEnv_t;
      typedef Cont_t                *PCont_t;
      typedef Value_t               *PValue_t;

      virtual ~Type() {}

      static inline PCont_t object(void* ptr)   {
         return PCont_t(PEnv_t(ptr)->fObject);
      }
      static void* size(void* env)  {
         PEnv_t  e = PEnv_t(env);
         e->fSize   = PCont_t(e->fObject)->size();
         return &e->fSize;
      }
      static void* clear(void* env)  {
         object(env)->clear();
         return 0;
      }
      static void* first(void* env)  {
         PEnv_t  e = PEnv_t(env);
         PCont_t c = PCont_t(e->fObject);
#if 0
         // Assume iterators do not need destruction
         ::new(e->buff) Iter_t(c->begin());
#endif
         e->fIterator = c->begin();
         e->fSize  = c->size();
         if ( 0 == e->fSize ) return e->fStart = 0;
         TYPENAME T::const_reference ref = *(e->iter());
         return e->fStart = Type<T>::address(ref);
      }
      static void* next(void* env)  {
         PEnv_t  e = PEnv_t(env);
         PCont_t c = PCont_t(e->fObject);
         for (; e->fIdx > 0 && e->iter() != c->end(); ++(e->iter()), --e->fIdx){ }
         // TODO: Need to find something for going backwards....
         if ( e->iter() == c->end() ) return 0;
         TYPENAME T::const_reference ref = *(e->iter());
         return Type<T>::address(ref);
      }
      static void* construct(void *what, size_t size)  {
         PValue_t m = PValue_t(what);
         for (size_t i=0; i<size; ++i, ++m)
            ::new(m) Value_t();
         return 0;
      }
      static void* collect(void *coll, void *array)  {
         PCont_t  c = PCont_t(coll);
         PValue_t m = PValue_t(array);
         for (Iter_t i=c->begin(); i != c->end(); ++i, ++m )
            ::new(m) Value_t(*i);
         return 0;
      }
      static void destruct(void *what, size_t size)  {
         PValue_t m = PValue_t(what);
         for (size_t i=0; i < size; ++i, ++m )
            m->~Value_t();
      }

      static const bool fgLargeIterator = sizeof(typename Cont_t::iterator) > fgIteratorArenaSize;
      typedef Iterators<Cont_t,fgLargeIterator> Iterators_t;

   };

   /** @class TCollectionProxyInfo::Map TCollectionProxyInfo.h TCollectionProxyInfo.h
    *
    * Small helper to encapsulate all necessary data accesses for
    * containers like vector, list, deque
    *
    * @author  M.Frank
    * @version 1.0
    * @date    10/10/2004
    */
   template <class T> struct Pushback : public Type<T> {
      typedef T                      Cont_t;
      typedef typename T::iterator   Iter_t;
      typedef typename T::value_type Value_t;
      typedef Environ<Iter_t>        Env_t;
      typedef Env_t                 *PEnv_t;
      typedef Cont_t                *PCont_t;
      typedef Value_t               *PValue_t;
      static void resize(void* obj, size_t n) {
         PCont_t c = PCont_t(obj);
         c->resize(n);
      }
      static void* feed(void *from, void *to, size_t size)  {
         PCont_t  c = PCont_t(to);
         PValue_t m = PValue_t(from);
         for (size_t i=0; i<size; ++i, ++m)
            c->push_back(*m);
         return 0;
      }
      static int value_offset()  {
         return 0;
      }
   };

   /** @class TCollectionProxyInfo::Map TCollectionProxyInfo.h TCollectionProxyInfo.h
    *
    * Small helper to encapsulate all necessary data accesses for
    * containers like set, multiset etc.
    *
    * @author  M.Frank
    * @version 1.0
    * @date    10/10/2004
    */
   template <class T> struct Insert : public Type<T> {
      typedef T                      Cont_t;
      typedef typename T::iterator   Iter_t;
      typedef typename T::value_type Value_t;
      typedef Environ<Iter_t>        Env_t;
      typedef Env_t                 *PEnv_t;
      typedef Cont_t                *PCont_t;
      typedef Value_t               *PValue_t;
      static void* feed(void *from, void *to, size_t size)  {
         PCont_t  c = PCont_t(to);
         PValue_t m = PValue_t(from);
         for (size_t i=0; i<size; ++i, ++m)
            c->insert(*m);
         return 0;
      }
      static void resize(void* /* obj */, size_t )  {
         ;
      }
      static int value_offset()  {
         return 0;
      }
   };

   /** @class TCollectionProxyInfo::Map TCollectionProxyInfo.h TCollectionProxyInfo.h
    *
    * Small helper to encapsulate all necessary data accesses for
    * containers like set, multiset etc.
    *
    * @author  M.Frank
    * @version 1.0
    * @date    10/10/2004
    */
   template <class T> struct MapInsert : public Type<T> {
      typedef T                      Cont_t;
      typedef typename T::iterator   Iter_t;
      typedef typename T::value_type Value_t;
      typedef Environ<Iter_t>        Env_t;
      typedef Env_t                 *PEnv_t;
      typedef Cont_t                *PCont_t;
      typedef Value_t               *PValue_t;
      static void* feed(void *from, void *to, size_t size)  {
         PCont_t  c = PCont_t(to);
         PValue_t m = PValue_t(from);
         for (size_t i=0; i<size; ++i, ++m)
            c->insert(*m);
         return 0;
      }
      static void resize(void* /* obj */, size_t )  {
         ;
      }
      static int value_offset()  {
         return ((char*)&((PValue_t(0x1000))->second)) - ((char*)PValue_t(0x1000));
      }
   };

      
   public:
#ifndef __CINT__
      const type_info &fInfo;
#endif
      size_t fIterSize;
      size_t fValueDiff;
      int    fValueOffset;
      void*  (*fSizeFunc)(void*);
      void   (*fResizeFunc)(void*,size_t);
      void*  (*fClearFunc)(void*);
      void*  (*fFirstFunc)(void*);
      void*  (*fNextFunc)(void*);
      void*  (*fConstructFunc)(void*,size_t);
      void   (*fDestructFunc)(void*,size_t);
      void*  (*fFeedFunc)(void*,void*,size_t);
      void*  (*fCollectFunc)(void*,void*);
      void*  (*fCreateEnv)();
      
      // Set of function of direct iteration of the collections.
      void (*fCreateIterators)(void *collection, void **begin_arena, void **end_arena, TVirtualCollectionProxy *proxy);
      // begin_arena and end_arena should contain the location of memory arena  of size fgIteratorSize. 
      // If the collection iterator are of that size or less, the iterators will be constructed in place in those location (new with placement)
      // Otherwise the iterators will be allocated via a regular new and their address returned by modifying the value of begin_arena and end_arena.
      
      void* (*fCopyIterator)(void *dest, const void *source);
      // Copy the iterator source, into dest.   dest should contain should contain the location of memory arena  of size fgIteratorSize.
      // If the collection iterator are of that size or less, the iterator will be constructed in place in this location (new with placement)
      // Otherwise the iterator will be allocated via a regular new and its address returned by modifying the value of dest.
      
      void* (*fNext)(void *iter, const void *end);
      // iter and end should be pointer to respectively an iterator to be incremented and the result of colleciton.end()
      // 'Next' will increment the iterator 'iter' and return 0 if the iterator reached the end.
      // If the end is not reached, 'Next' will return the address of the content unless the collection contains pointers in
      // which case 'Next' will return the value of the pointer.
      
      void (*fDeleteSingleIterator)(void *iter);
      void (*fDeleteTwoIterators)(void *begin, void *end);
      // If the sizeof iterator is greater than fgIteratorArenaSize, call delete on the addresses,
      // Otherwise just call the iterator's destructor.
      
   public:
      TCollectionProxyInfo(const type_info& info,
                           size_t iter_size,
                           size_t value_diff,
                           int    value_offset,
                           void*  (*size_func)(void*),
                           void   (*resize_func)(void*,size_t),
                           void*  (*clear_func)(void*),
                           void*  (*first_func)(void*),
                           void*  (*next_func)(void*),
                           void*  (*construct_func)(void*,size_t),
                           void   (*destruct_func)(void*,size_t),
                           void*  (*feed_func)(void*,void*,size_t),
                           void*  (*collect_func)(void*,void*),
                           void*  (*create_env)(),
                           void   (*getIterators)(void *collection, void **begin_arena, void **end_arena, TVirtualCollectionProxy *proxy) = 0,
                           void*  (*copyIterator)(void *dest, const void *source) = 0,
                           void*  (*next)(void *iter, const void *end) = 0,
                           void   (*deleteSingleIterator)(void *iter) = 0,
                           void   (*deleteTwoIterators)(void *begin, void *end) = 0
                           ) :
         fInfo(info), fIterSize(iter_size), fValueDiff(value_diff),
         fValueOffset(value_offset),
         fSizeFunc(size_func),fResizeFunc(resize_func),fClearFunc(clear_func),
         fFirstFunc(first_func),fNextFunc(next_func),fConstructFunc(construct_func),
         fDestructFunc(destruct_func),fFeedFunc(feed_func),fCollectFunc(collect_func),
         fCreateEnv(create_env),
         fCreateIterators(getIterators),fCopyIterator(copyIterator),fNext(next),
         fDeleteSingleIterator(deleteSingleIterator),fDeleteTwoIterators(deleteTwoIterators)
      {
      }
 
      /// Generate proxy from template
      template <class T> static ROOT::TCollectionProxyInfo* Generate(const T&)  {
         // Generate a TCollectionProxyInfo given a TCollectionProxyInfo::Type
         // template (used to described the behavior of the stl collection.
         // Typical use looks like:
         //      ::ROOT::TCollectionProxyInfo::Generate(TCollectionProxyInfo::Pushback< vector<string> >()));
         
         PairHolder<TYPENAME T::Value_t, TYPENAME T::Value_t>* p =
            (PairHolder<TYPENAME T::Value_t, TYPENAME T::Value_t>*)0x1000;
         return new ROOT::TCollectionProxyInfo(typeid(TYPENAME T::Cont_t),
                                               sizeof(TYPENAME T::Iter_t),
                                               (((char*)&p->second)-((char*)&p->first)),
                                               T::value_offset(),
                                               T::size,
                                               T::resize,
                                               T::clear,
                                               T::first,
                                               T::next,
                                               T::construct,
                                               T::destruct,
                                               T::feed,
                                               T::collect,
                                               T::Env_t::Create,
                                               T::Iterators_t::create,
                                               T::Iterators_t::copy,
                                               T::Iterators_t::next,
                                               T::Iterators_t::destruct1,
                                               T::Iterators_t::destruct2);
      }

      template <class T> static ROOT::TCollectionProxyInfo Get(const T&)  {

         // Generate a TCollectionProxyInfo given a TCollectionProxyInfo::Type
         // template (used to described the behavior of the stl collection.
         // Typical use looks like:
         //      ::ROOT::TCollectionProxyInfo::Get(TCollectionProxyInfo::Pushback< vector<string> >()));
         
         PairHolder<TYPENAME T::Value_t, TYPENAME T::Value_t>* p =
            (PairHolder<TYPENAME T::Value_t, TYPENAME T::Value_t>*)0x1000;
         return ROOT::TCollectionProxyInfo(typeid(TYPENAME T::Cont_t),
                                           sizeof(TYPENAME T::Iter_t),
                                           (((char*)&p->second)-((char*)&p->first)),
                                           T::value_offset(),
                                           T::size,
                                           T::resize,
                                           T::clear,
                                           T::first,
                                           T::next,
                                           T::construct,
                                           T::destruct,
                                           T::feed,
                                           T::collect,
                                           T::Env_t::Create);
      }

   };

   template <> struct TCollectionProxyInfo::Type<std::vector<Bool_t> >
   : public TCollectionProxyInfo::Address<std::vector<Bool_t>::const_reference>
   {
      typedef std::vector<Bool_t>             Cont_t;
      typedef std::vector<Bool_t>::iterator   Iter_t;
      typedef std::vector<Bool_t>::value_type Value_t;
      typedef Environ<Iter_t>                 Env_t;
      typedef Env_t                          *PEnv_t;
      typedef Cont_t                         *PCont_t;
      typedef Value_t                        *PValue_t;
      
      virtual ~Type() {}
      
      static inline PCont_t object(void* ptr)   {
         return PCont_t(PEnv_t(ptr)->fObject);
      }
      static void* size(void* env)  {
         PEnv_t  e = PEnv_t(env);
         e->fSize   = PCont_t(e->fObject)->size();
         return &e->fSize;
      }
      static void* clear(void* env)  {
         object(env)->clear();
         return 0;
      }
      static void* first(void* env)  {
         PEnv_t  e = PEnv_t(env);
         PCont_t c = PCont_t(e->fObject);
#if 0
         // Assume iterators do not need destruction
         ::new(e->buff) Iter_t(c->begin());
#endif
         e->fIterator = c->begin();
         e->fSize  = c->size();
         return 0;
      }
      static void* next(void* env)  {
         PEnv_t  e = PEnv_t(env);
         PCont_t c = PCont_t(e->fObject);
         for (; e->fIdx > 0 && e->iter() != c->end(); ++(e->iter()), --e->fIdx){ }
         // TODO: Need to find something for going backwards....
         return 0;
      }
      static void* construct(void*,size_t)  {
         // Nothing to construct.
         return 0;
      }
      static void* collect(void *coll, void *array)  {
         PCont_t  c = PCont_t(coll);
         PValue_t m = PValue_t(array); // 'start' is a buffer outside the container.
         for (Iter_t i=c->begin(); i != c->end(); ++i, ++m )
            ::new(m) Value_t(*i);
         return 0;
      }
      static void destruct(void*,size_t)  {
         // Nothing to destruct.
      }

      //static const bool fgLargeIterator = sizeof(Cont_t::iterator) > fgIteratorArenaSize;
      //typedef Iterators<Cont_t,fgLargeIterator> Iterators_t;

      struct Iterators {
         typedef Cont_t *PCont_t;
         typedef Cont_t::iterator iterator;

         static void create(void *coll, void **begin_arena, void **end_arena, TVirtualCollectionProxy*) {
            PCont_t c = PCont_t(coll);
            new (*begin_arena) iterator(c->begin());
            new (*end_arena) iterator(c->end());
         }
         static void* copy(void *dest_arena, const void *source_ptr) {
            const iterator *source = (const iterator *)(source_ptr);
            new (dest_arena) iterator(*source);
            return dest_arena;
         }
         static void* next(void *iter_loc, const void *end_loc) {
            const iterator *end = (const iterator *)(end_loc);
            iterator *iter = (iterator *)(iter_loc);
            if (*iter != *end) {
               ++(*iter);
               //if (*iter != *end) {
               //   return IteratorValue<Cont_t, Cont_t::value_type>::get(*iter);
               //}
            }
            return 0;
         }
         static void destruct1(void *iter_ptr) {
            iterator *start = (iterator *)(iter_ptr);
            start->~iterator();
         }
         static void destruct2(void *begin_ptr, void *end_ptr) {
            iterator *start = (iterator *)(begin_ptr);
            iterator *end = (iterator *)(end_ptr);
            start->~iterator();
            end->~iterator();
         }
      };
      typedef Iterators Iterators_t;

   };
   
   template <> struct TCollectionProxyInfo::Pushback<std::vector<bool> > : public TCollectionProxyInfo::Type<std::vector<Bool_t> > {
      typedef std::vector<Bool_t>    Cont_t;
      typedef Cont_t::iterator       Iter_t;
      typedef Cont_t::value_type     Value_t;
      typedef Environ<Iter_t>        Env_t;
      typedef Env_t                 *PEnv_t;
      typedef Cont_t                *PCont_t;
      typedef Value_t               *PValue_t;
      
      static void resize(void* obj,size_t n) {
         PCont_t c = PCont_t(obj);
         c->resize(n);
      }
      static void* feed(void* from, void *to, size_t size)  {
         PCont_t  c = PCont_t(to);
         PValue_t m = PValue_t(from);
         for (size_t i=0; i<size; ++i, ++m)
            c->push_back(*m);
         return 0;
      }
      static int value_offset()  {
         return 0;
      }
   };

#ifndef __CINT__
   // Need specialization for boolean references due to stupid STL vector<bool>
   template<> inline void* ::ROOT::TCollectionProxyInfo::Address<std::vector<Bool_t>::const_reference>::address(std::vector<Bool_t>::const_reference ) {
      R__ASSERT(0);
      return 0;
   }
#endif
   
   template <typename T> class TStdBitsetHelper {
      // This class is intentionally empty, this is scaffolding to allow the equivalent
      // of 'template <int N> struct TCollectionProxyInfo::Type<std::bitset<N> >' which 
      // is not effective in C++ (as of gcc 4.3.3).
   };

#ifndef __CINT__
   template <typename Bitset_t> struct TCollectionProxyInfo::Type<ROOT::TStdBitsetHelper<Bitset_t> > : public TCollectionProxyInfo::Address<const Bool_t &>
   {
      typedef Bitset_t                 Cont_t;
      typedef std::pair<size_t,Bool_t> Iter_t;
      typedef Bool_t                   Value_t;
      typedef Environ<Iter_t>          Env_t;
      typedef Env_t                   *PEnv_t;
      typedef Cont_t                  *PCont_t;
      typedef Value_t                 *PValue_t;
      
      virtual ~Type() {}
      
      static inline PCont_t object(void* ptr)   {
         return PCont_t(PEnv_t(ptr)->fObject);
      }
      static void* size(void* env)  {
         PEnv_t  e = PEnv_t(env);
         e->fSize   = PCont_t(e->fObject)->size();
         return &e->fSize;
      }
      static void* clear(void* env)  {
         object(env)->reset();
         return 0;
      }
      static void* first(void* env)  {
         PEnv_t  e = PEnv_t(env);
         PCont_t c = PCont_t(e->fObject);
         e->fIterator.first = 0;
         e->fIterator.second = c->size() > 0 ? c->test(e->fIterator.first) : false ;  // Iterator actually hold the value.
         e->fSize  = c->size();
         return &(e->fIterator.second);
      }
      static void* next(void* env)  {
         PEnv_t  e = PEnv_t(env);
         PCont_t c = PCont_t(e->fObject);
         for (; e->fIdx > 0 && e->fIterator.first != c->size(); ++(e->fIterator.first), --e->fIdx){ }
         e->fIterator.second = (e->fIterator.first != c->size()) ? c->test(e->fIterator.first) : false;
         return &(e->fIterator.second);
      }
      static void* construct(void*,size_t)  {
         // Nothing to construct.
         return 0;
      }
      static void* collect(void *coll, void *array)  {
         PCont_t  c = PCont_t(coll);
         PValue_t m = PValue_t(array); // 'start' is a buffer outside the container.
         for (size_t i=0; i != c->size(); ++i, ++m )
            *m = c->test(i);
         return 0;
      }
      static void destruct(void*,size_t)  {
         // Nothing to destruct.
      }

      //static const bool fgLargeIterator = sizeof(typename Cont_t::iterator) > fgIteratorArenaSize;
      //typedef Iterators<Cont_t,fgLargeIterator> Iterators_t;

      struct Iterators {
         typedef Cont_t *PCont_t;
         union PtrSize_t { size_t fIndex; void *fAddress; };
         typedef std::pair<PtrSize_t,Bool_t> iterator;
         // In the end iterator we store the bitset pointer
         // and do not use the 'second' part of the pair.
         // In the other iterator we store the index
         // and the value.

         static void create(void *coll, void **begin_arena, void **end_arena, TVirtualCollectionProxy*) {
            iterator *begin = new (*begin_arena) iterator;
            begin->first.fIndex = 0;
            begin->second = false;
            iterator *end = new (*end_arena) iterator;
            end->first.fAddress = coll;
            end->second = false;
         }
         static void* copy(void *dest_arena, const void *source_ptr) {
            const iterator *source = (const iterator *)(source_ptr);
            new (dest_arena) iterator(*source);
            return dest_arena;
         }
         static void* next(void *iter_loc, const void *end_loc) {
            const iterator *end = (const iterator *)(end_loc);
            PCont_t c = (PCont_t)end->first.fAddress;
            iterator *iter = (iterator *)(iter_loc);
            if (iter->first.fIndex != c->size()) {
               iter->second = c->test(iter->first.fIndex);
               ++(iter->first.fIndex);
            }
            return &(iter->second);
         }
         static void destruct1(void *iter_ptr) {
            iterator *start = (iterator *)(iter_ptr);
            start->~iterator();
         }
         static void destruct2(void *begin_ptr, void *end_ptr) {
            iterator *start = (iterator *)(begin_ptr);
            iterator *end = (iterator *)(end_ptr);
            start->~iterator();
            end->~iterator();
         }
      };
      typedef Iterators Iterators_t;
   };
   
   template <typename Bitset_t> 
   struct TCollectionProxyInfo::Pushback<ROOT::TStdBitsetHelper<Bitset_t>  > : public TCollectionProxyInfo::Type<TStdBitsetHelper<Bitset_t> > {
      typedef Bitset_t         Cont_t;
      typedef bool             Iter_t;
      typedef bool             Value_t;
      typedef Environ<Iter_t>  Env_t;
      typedef Env_t           *PEnv_t;
      typedef Cont_t          *PCont_t;
      typedef Value_t         *PValue_t;
      
      static void resize(void*,size_t)  {
      }
      static void* feed(void *from, void *to, size_t size)  {
         PCont_t  c = PCont_t(to);
         PValue_t m = PValue_t(from); 
         for (size_t i=0; i<size; ++i, ++m)
            c->set(i,*m);
         return 0;
      }
      static int value_offset()  {
         return 0;
      }
   };
#endif
   
}

#endif
