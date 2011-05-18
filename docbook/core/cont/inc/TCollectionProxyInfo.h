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
      Bool_t              fUseTemp;
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
      static void* collect(void* env)  {
         PEnv_t   e = PEnv_t(env);
         PCont_t  c = PCont_t(e->fObject);
         PValue_t m = PValue_t(e->fStart);
         for (Iter_t i=c->begin(); i != c->end(); ++i, ++m )
            ::new(m) Value_t(*i);
         return 0;
      }
      static void destruct(void *what, size_t size)  {
         PValue_t m = PValue_t(what);
         for (size_t i=0; i < size; ++i, ++m )
            m->~Value_t();
      }
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
      void*  (*fCollectFunc)(void*);
      void*  (*fCreateEnv)();
      
      // Set of function of direct iteration of the collections.
      void (*fGetIterators)(void *collection, void *&begin_arena, void *&end_arena); 
      // begin_arena and end_arena should contain the location of memory arena  of size fgIteratorSize. 
      // If the collection iterator are of that size or less, the iterators will be constructed in place in those location (new with placement)
      // Otherwise the iterators will be allocated via a regular new and their address returned by modifying the value of begin_arena and end_arena.
      
      void (*fCopyIterator)(void *&dest, const void *source);
      // Copy the iterator source, into dest.   dest should contain should contain the location of memory arena  of size fgIteratorSize.
      // If the collection iterator are of that size or less, the iterator will be constructed in place in this location (new with placement)
      // Otherwise the iterator will be allocated via a regular new and its address returned by modifying the value of dest.
      
      void* (*fNext)(void *iter, void *end);
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
                           void*  (*collect_func)(void*),
                           void*  (*create_env)()
                           ) :
         fInfo(info), fIterSize(iter_size), fValueDiff(value_diff),
         fValueOffset(value_offset),
         fSizeFunc(size_func),fResizeFunc(resize_func),fClearFunc(clear_func),
         fFirstFunc(first_func),fNextFunc(next_func),fConstructFunc(construct_func),
         fDestructFunc(destruct_func),fFeedFunc(feed_func),fCollectFunc(collect_func),
         fCreateEnv(create_env),
         fGetIterators(0),fCopyIterator(0),fNext(0),fDeleteSingleIterator(0),fDeleteTwoIterators(0)
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
                                               T::Env_t::Create);
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

   template <> struct TCollectionProxyInfo::Type<std::vector<bool> >
   : public TCollectionProxyInfo::Address<std::vector<bool>::const_reference>
   {
      typedef std::vector<bool>             Cont_t;
      typedef std::vector<bool>::iterator   Iter_t;
      typedef std::vector<bool>::value_type Value_t;
      typedef Environ<Iter_t>               Env_t;
      typedef Env_t                        *PEnv_t;
      typedef Cont_t                       *PCont_t;
      typedef Value_t                      *PValue_t;
      
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
      static void* collect(void* env)  {
         PEnv_t   e = PEnv_t(env);
         PCont_t  c = PCont_t(e->fObject);
         PValue_t m = PValue_t(e->fStart); // 'start' is a buffer outside the container.
         for (Iter_t i=c->begin(); i != c->end(); ++i, ++m )
            ::new(m) Value_t(*i);
         return 0;
      }
      static void destruct(void*,size_t)  {
         // Nothing to destruct.
      }
   };
   
   template <> struct TCollectionProxyInfo::Pushback<std::vector<bool> > : public TCollectionProxyInfo::Type<std::vector<bool> > {
      typedef std::vector<bool>      Cont_t;
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
   template<> inline void* ::ROOT::TCollectionProxyInfo::Address<std::vector<bool>::const_reference>::address(std::vector<bool>::const_reference ) {
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
   template <typename Bitset_t> struct TCollectionProxyInfo::Type<ROOT::TStdBitsetHelper<Bitset_t> > : public TCollectionProxyInfo::Address<const bool &>
   {
      typedef Bitset_t                Cont_t;
      typedef std::pair<size_t,bool>  Iter_t;
      typedef bool                    Value_t;
      typedef Environ<Iter_t>         Env_t;
      typedef Env_t                  *PEnv_t;
      typedef Cont_t                 *PCont_t;
      typedef Value_t                *PValue_t;
      
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
         return 0;
      }
      static void* next(void* env)  {
         PEnv_t  e = PEnv_t(env);
         PCont_t c = PCont_t(e->fObject);
         for (; e->fIdx > 0 && e->fIterator.first != c->size(); ++(e->fIterator.first), --e->fIdx){ }
         e->fIterator.second = (e->fIterator.first != c->size()) ? c->test(e->fIterator.first) : false;
         return 0;
      }
      static void* construct(void*,size_t)  {
         // Nothing to construct.
         return 0;
      }
      static void* collect(void* env)  {
         PEnv_t   e = PEnv_t(env);
         PCont_t  c = PCont_t(e->fObject);
         PValue_t m = PValue_t(e->fStart); // 'start' is a buffer outside the container.
         for (size_t i=0; i != c->size(); ++i, ++m )
            *m = c->test(i);
         return 0;
      }
      static void destruct(void*,size_t)  {
         // Nothing to destruct.
      }
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
