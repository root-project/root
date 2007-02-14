// @(#)root/cont:$Name:  $:$Id: TCollectionProxy.h,v 1.17 2007/01/16 14:31:49 brun Exp $
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
   template <typename T> struct Environ  {
      typedef T           Iter_t;
      char                buff[64];
      size_t              idx;
      size_t              size;
      void*               object;
      void*               start;
      void*               temp;
      Bool_t              delete_temp;
      int                 refCount;
      size_t              space;
      T& iter() { return *(T*)buff; }
   };
#else
   template <typename T> struct Environ;
#endif
#if defined(R__VCXX6)
   template <class T> void Destruct(T* obj) { obj->~T(); }
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
#ifdef R__KCC
      : public Address<TYPENAME T::value_type&>
#else
      : public Address<TYPENAME T::const_reference>
#endif
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
         return PCont_t(PEnv_t(ptr)->object);
      }
      static void* size(void* env)  {
         PEnv_t  e = PEnv_t(env);
         e->size   = PCont_t(e->object)->size();
         return &e->size;
      }
      static void* clear(void* env)  {
         object(env)->clear();
         return 0;
      }
      static void* first(void* env)  {
         PEnv_t  e = PEnv_t(env);
         PCont_t c = PCont_t(e->object);
         // Assume iterators do not need destruction
         ::new(e->buff) Iter_t(c->begin());
         e->size  = c->size();
         if ( 0 == e->size ) return e->start = 0;
#ifdef R__KCC
         TYPENAME T::value_type& ref = *(e->iter());
#else
         TYPENAME T::const_reference ref = *(e->iter());
#endif
         return e->start = address(ref);
      }
      static void* next(void* env)  {
         PEnv_t  e = PEnv_t(env);
         PCont_t c = PCont_t(e->object);
         for (; e->idx > 0 && e->iter() != c->end(); ++(e->iter()), --e->idx );
         // TODO: Need to find something for going backwards....
         if ( e->iter() == c->end() ) return 0;
#ifdef R__KCC
         TYPENAME T::value_type& ref = *(e->iter());
#else
         TYPENAME T::const_reference ref = *(e->iter());
#endif
         return address(ref);
      }
      static void* construct(void* env)  {
         PEnv_t  e = PEnv_t(env);
         PValue_t m = PValue_t(e->start);
         for (size_t i=0; i<e->size; ++i, ++m)
            ::new(m) Value_t();
         return 0;
      }
      static void* collect(void* env)  {
         PEnv_t   e = PEnv_t(env);
         PCont_t  c = PCont_t(e->object);
         PValue_t m = PValue_t(e->start);
         for (Iter_t i=c->begin(); i != c->end(); ++i, ++m )
            ::new(m) Value_t(*i);
         return 0;
      }
      static void* destruct(void* env)  {
         PEnv_t   e = PEnv_t(env);
         PValue_t m = PValue_t(e->start);
#if defined(R__VCXX6)
         PCont_t  c = PCont_t(e->object);
         for (size_t i=0; i < e->size; ++i, ++m )
            ROOT::Destruct(m);
#else
         for (size_t i=0; i < e->size; ++i, ++m )
            m->~Value_t();
#endif
         return 0;
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
      static void* resize(void* env)  {
         PEnv_t  e = PEnv_t(env);
         PCont_t c = PCont_t(e->object);
         c->resize(e->size);
         e->idx = 0;
         return e->start = address(*c->begin());
      }
      static void* feed(void* env)  {
         PEnv_t   e = PEnv_t(env);
         PCont_t  c = PCont_t(e->object);
         PValue_t m = PValue_t(e->start);
         for (size_t i=0; i<e->size; ++i, ++m)
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
      static void* feed(void* env)  {
         PEnv_t   e = PEnv_t(env);
         PCont_t  c = PCont_t(e->object);
         PValue_t m = PValue_t(e->start);
         for (size_t i=0; i<e->size; ++i, ++m)
            c->insert(*m);
         return 0;
      }
      static void* resize(void* /* env */ )  {
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
   template <class T> struct MapInsert : public Type<T> {
      typedef T                      Cont_t;
      typedef typename T::iterator   Iter_t;
      typedef typename T::value_type Value_t;
      typedef Environ<Iter_t>        Env_t;
      typedef Env_t                 *PEnv_t;
      typedef Cont_t                *PCont_t;
      typedef Value_t               *PValue_t;
      static void* feed(void* env)  {
         PEnv_t   e = PEnv_t(env);
         PCont_t  c = PCont_t(e->object);
         PValue_t m = PValue_t(e->start);
         for (size_t i=0; i<e->size; ++i, ++m)
            c->insert(*m);
         return 0;
      }
      static void* resize(void* /* env */ )  {
         return 0;
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
      void*  (*fResizeFunc)(void*);
      void*  (*fClearFunc)(void*);
      void*  (*fFirstFunc)(void*);
      void*  (*fNextFunc)(void*);
      void*  (*fConstructFunc)(void*);
      void*  (*fDestructFunc)(void*);
      void*  (*fFeedFunc)(void*);
      void*  (*fCollectFunc)(void*);

   public:
      TCollectionProxyInfo(const type_info& info,
                           size_t iter_size,
                           size_t value_diff,
                           int    value_offset,
                           void*  (*size_func)(void*),
                           void*  (*resize_func)(void*),
                           void*  (*clear_func)(void*),
                           void*  (*first_func)(void*),
                           void*  (*next_func)(void*),
                           void*  (*construct_func)(void*),
                           void*  (*destruct_func)(void*),
                           void*  (*feed_func)(void*),
                           void*  (*collect_func)(void*)
                           ) :
         fInfo(info), fIterSize(iter_size), fValueDiff(value_diff),
         fValueOffset(value_offset),
         fSizeFunc(size_func),fResizeFunc(resize_func),fClearFunc(clear_func),
         fFirstFunc(first_func),fNextFunc(next_func),fConstructFunc(construct_func),
         fDestructFunc(destruct_func),fFeedFunc(feed_func),fCollectFunc(collect_func)
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
                                               T::collect);
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
                                           T::collect);
      }

   };

#ifndef __CINT__
   // Need specialization for boolean references due to stupid STL vector<bool>
   template<> inline void* ::ROOT::TCollectionProxyInfo::Address<std::vector<bool>::const_reference>::address(std::vector<bool>::const_reference ) {
      return 0;
   }
#endif

}

#endif
