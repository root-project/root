// @(#)root/reflex:$Id$
// Author: Markus Frank 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_CollectionProxy
#define Reflex_CollectionProxy 1 1

#include <cstddef>

// Macro indicating the version of the Collection Proxy interface followed
// by this Reflex build, this must match the version number of
// ROOT_COLLECTIONPROXY_VERSION in ROOT's TVirtutalCollectionProxy.h

#define REFLEX_COLLECTIONPROXY_VERSION 3

// Forward declarations
namespace std {
template <class T, class A> class deque;
template <class T, class A> class vector;
template <class T, class A> class list;
template <class T, class A> class queue;
template <class T, class A> class stack;
template <class K, class T, class A> class set;
template <class K, class T, class A> class multiset;
template <class K, class T, class R, class A> class map;
template <class K, class T, class R, class A> class multimap;
template <class T> class allocator;
}
// Hash map forward declarations
#if defined(__GNUC__)
namespace __gnu_cxx {  // GNU GCC
template <class T, class F, class E, class A> class hash_set;
template <class T, class F, class E, class A> class hash_multiset;
template <class K, class T, class F, class E, class A> class hash_map;
template <class K, class T, class F, class E, class A> class hash_multimap;
}
#elif  defined(_WIN32)
namespace stdext {     // Visual C++
template <class K, class T, class A> class hash_set;
template <class K, class T, class A> class hash_multiset;
template <class K, class T, class R, class A> class hash_map;
template <class K, class T, class R, class A> class hash_multimap;
}
#endif

namespace Reflex  {
#ifndef __CINT__
struct EnvironBase {
   EnvironBase(): fIdx(0),
      fSize(0),
      fObject(0),
      fStart(0),
      fTemp(0),
      fDeleteTemp(false),
      fRefSize(1),
      fSpace(0) {
      //   fprintf("Running default constructor on %p\n",this);
   }


   virtual ~EnvironBase() {}

   size_t fIdx;
   size_t fSize;
   void* fObject;
   void* fStart;
   void* fTemp;
   bool fDeleteTemp;
   int fRefSize;
   size_t fSpace;
};
template <typename T> struct Environ: public EnvironBase {
   typedef T Iter_t;
   Iter_t fIterator;
   T&
   iter() { return fIterator; }

   static void*
   Create() {
      return new Environ();
   }


};
#else
struct EnvironBase;
template <typename T> struct Environ;
#endif

template <typename T> struct Address {
   static void*
   address(T ref) {
      return (void*)& ref;
   }


};

template <class T> struct CollType
#ifdef _KCC  // KAI compiler
   : public Address<typename T::value_type&>
#else
   : public Address<typename T::const_reference>
#endif
{
#ifdef _KCC  // KAI compiler
   typedef Address<typename T::value_type&> Address_t;
#else
   typedef Address<typename T::const_reference> Address_t;
#endif
   typedef T Cont_t;
   typedef typename T::iterator Iter_t;
   typedef typename T::value_type Value_t;
   typedef Reflex::Environ<Iter_t> Env_t;
   typedef Env_t* PEnv_t;
   typedef Cont_t* PCont_t;
   typedef Value_t* PValue_t;

   static inline PCont_t
   object(void* ptr) {
      return PCont_t(PEnv_t(ptr)->fObject);
   }


   static void*
   size(void* env) {
      PEnv_t e = PEnv_t(env);
      e->fSize = PCont_t(e->fObject)->size();
      return &e->fSize;
   }


   static void*
   clear(void* env) {
      object(env)->clear();
      return 0;
   }


   static void*
   first(void* env) {
      PEnv_t e = PEnv_t(env);
      PCont_t c = PCont_t(e->fObject);
      // Assume iterators do not need destruction
      e->fIterator = c->begin();
      e->fSize = c->size();

      if (0 == e->fSize) { return e->fStart = 0; }
#ifdef _KCC  // KAI compiler
      typename T::value_type& ref = *(e->iter());
#else
      typename T::const_reference ref = *(e->iter());
#endif
      return e->fStart = Address_t::address(ref);
   }


   static void*
   next(void* env) {
      PEnv_t e = PEnv_t(env);
      PCont_t c = PCont_t(e->fObject);

      for ( ; e->fIdx > 0 && e->iter() != c->end(); ++(e->iter()), --e->fIdx) {}

      // TODO: Need to find something for going backwards....
      if (e->iter() == c->end()) { return 0; }
#ifdef _KCC  // KAI compiler
      typename T::value_type& ref = *(e->iter());
#else
      typename T::const_reference ref = *(e->iter());
#endif
      return Address_t::address(ref);
   }


   static void*
   construct(void* what, size_t size) {
      PValue_t m = PValue_t(what);

      for (size_t i = 0; i < size; ++i, ++m) {
         ::new (m) Value_t();
      }
      return 0;
   }


   static void*
   collect(void* env) {
      PEnv_t e = PEnv_t(env);
      PCont_t c = PCont_t(e->fObject);
      PValue_t m = PValue_t(e->fStart);

      for (Iter_t i = c->begin(); i != c->end(); ++i, ++m) {
         ::new (m) Value_t(*i);
      }
      return 0;
   }


   static void
   destruct(void* what, size_t size) {
      PValue_t m = PValue_t(what);

      for (size_t i = 0; i < size; ++i, ++m) {
         m->~Value_t();
      }
   }


};

/** @class TCollectionProxy::Map TCollectionProxy.h TCollectionProxy.h
 *
 * Small helper to encapsulate all necessary data accesses for
 * containers like vector, list, deque
 *
 * @author  M.Frank
 * @version 1.0
 * @date    10/10/2004
 */
template <class T> struct Pushback: public CollType<T> {
   typedef T Cont_t;
   typedef typename T::iterator Iter_t;
   typedef typename T::value_type Value_t;
   typedef Environ<Iter_t> Env_t;
   typedef Env_t* PEnv_t;
   typedef Cont_t* PCont_t;
   typedef Value_t* PValue_t;
   static void resize(void* obj, size_t n) {
      PCont_t c = PCont_t(obj);
      c->resize(n);
   }

   static void*
      feed(void*from,void *to,size_t size) {
      PValue_t m = PValue_t(from);
      PCont_t c  = PCont_t(to);

      for (size_t i = 0; i < size; ++i, ++m) {
         c->push_back(*m);
      }
      return 0;
   }


   static int
   value_offset() {
      return 0;
   }


};

/** @class TCollectionProxy::Map TCollectionProxy.h TCollectionProxy.h
 *
 * Small helper to encapsulate all necessary data accesses for
 * containers like set, multiset etc.
 *
 * @author  M.Frank
 * @version 1.0
 * @date    10/10/2004
 */
template <class T> struct Insert: public CollType<T> {
   typedef T Cont_t;
   typedef typename T::iterator Iter_t;
   typedef typename T::value_type Value_t;
   typedef Environ<Iter_t> Env_t;
   typedef Env_t* PEnv_t;
   typedef Cont_t* PCont_t;
   typedef Value_t* PValue_t;

   static void*
      feed(void*from,void*to,size_t size) {
      PValue_t m = PValue_t(from);
      PCont_t c  = PCont_t(to);

      for (size_t i = 0; i < size; ++i, ++m) {
         c->insert(*m);
      }
      return 0;
   }


   static void resize(void* /* obj */, size_t) { }


   static int
   value_offset() {
      return 0;
   }


};

/** @class TCollectionProxy::Map TCollectionProxy.h TCollectionProxy.h
 *
 * Small helper to encapsulate all necessary data accesses for
 * containers like Set, multiset etc.
 *
 * @author  M.Frank
 * @version 1.0
 * @date    10/10/2004
 */
template <class T> struct MapInsert: public CollType<T> {
   typedef T Cont_t;
   typedef typename T::iterator Iter_t;
   typedef typename T::value_type Value_t;
   typedef Environ<Iter_t> Env_t;
   typedef Env_t* PEnv_t;
   typedef Cont_t* PCont_t;
   typedef Value_t* PValue_t;

   static void*
      feed(void*from,void *to,size_t size) {
      PValue_t m = PValue_t(from);
      PCont_t  c = PCont_t(to);

      for (size_t i = 0; i < size; ++i, ++m) {
         c->insert(*m);
      }
      return 0;
   }


   static void resize(void* /* obj */, size_t) {
   }


   static int
   value_offset() {
      return ((char*) &((PValue_t(0x1000))->second)) - ((char*) PValue_t(0x1000));
   }


};

#ifndef __CINT__
// Need specialization for boolean references due to stupid STL vector<bool>
template <> inline void* Reflex::Address<std::vector<bool, std::allocator<bool> >::const_reference
>::address(std::vector<bool, std::allocator<bool> >::const_reference) {
   return 0;
}


#endif

}

#include <vector>
namespace Reflex  {
/** @class CollFuncTable
 *
 * Table containing pointers to concrete functions to manipulate
 * Collections in a generic way
 *
 * @author  M.Frank
 */
struct RFLX_API CollFuncTable  {
   size_t iter_size;
   size_t value_diff;
   int value_offset;
   void*  (*size_func)(void*);
   void   (*resize_func)(void*,size_t);
   void*  (*clear_func)(void*);
   void*  (*first_func)(void*);
   void*  (*next_func)(void*);
   void*  (*construct_func)(void*,size_t);
   void   (*destruct_func)(void*,size_t);
   void*  (*feed_func)(void*,void*,size_t);
   void*  (*collect_func)(void*);
   void*  (*create_env)();
};

template <typename T> struct CFTGenerator {
   static CollFuncTable*
   Generate() {
      typedef typename T::Value_t Value_t;
      typedef std::pair<Value_t, Value_t> Pair_t;
      Pair_t* ptr = (Pair_t*) 0x1000;
      CollFuncTable* p = new CollFuncTable();
      p->iter_size = sizeof(typename T::Iter_t);
      p->value_diff = ((char*) &ptr->second) - ((char*) &ptr->first);
      p->value_offset = T::value_offset();
      p->size_func = T::size;
      p->first_func = T::first;
      p->next_func = T::next;
      p->clear_func = T::clear;
      p->resize_func = T::resize;
      p->collect_func = T::collect;
      p->construct_func = T::construct;
      p->destruct_func = T::destruct;
      p->feed_func = T::feed;
      p->create_env = T::Env_t::Create;
      return p;
   } // Generate


};
struct CFTNullGenerator {
   static void*
   Void_func(void*) {
      return 0;
   }


   static void*
   Void_func0() { return 0; }

   static void
   Void_func2b(void*,size_t) { ; }

   static void*
   Void_func2(void*,size_t) { return 0; }

   static void*
   Void_func3(void*,void*,size_t) { return 0; }

   static CollFuncTable*
   Generate() {
      CollFuncTable* p = new CollFuncTable();
      p->iter_size = 4;
      p->value_diff = 0;
      p->value_offset = 0;
      p->size_func = Void_func;
      p->first_func = Void_func;
      p->next_func = Void_func;
      p->clear_func = Void_func;
      p->resize_func = Void_func2b;
      p->collect_func = Void_func;
      p->construct_func = Void_func2;
      p->destruct_func = Void_func2b;
      p->feed_func = Void_func3;
      p->create_env = Void_func0;
      return p;
   } // Generate


};
// General proxy (dummy)
template <typename A> struct Proxy {};

// Specialization for std::vector
template <class T, class A> struct Proxy<std::vector<T, A> > {
   static CollFuncTable*
   Generate() {
      return CFTGenerator<Pushback<std::vector<T, A> > >::Generate();
   }


};
// Specialization for std::list
template <class T, class A> struct Proxy<std::list<T, A> > {
   static CollFuncTable*
   Generate() {
      return CFTGenerator<Pushback<std::list<T, A> > >::Generate();
   }


};
// Specialization for std::deque
template <class T, class A> struct Proxy<std::deque<T, A> > {
   static CollFuncTable*
   Generate() {
      return CFTGenerator<Pushback<std::deque<T, A> > >::Generate();
   }


};
// Specialization for std::set
template <class K, class T, class A> struct Proxy<std::set<K, T, A> > {
   static CollFuncTable*
   Generate() {
      return CFTGenerator<Insert<std::set<K, T, A> > >::Generate();
   }


};
// Specialization for std::multiset
template <class K, class T, class A> struct Proxy<std::multiset<K, T, A> > {
   static CollFuncTable*
   Generate() {
      return CFTGenerator<Insert<std::multiset<K, T, A> > >::Generate();
   }


};
// Specialization for std::map
template <class K, class T, class R, class A> struct Proxy<std::map<K, T, R, A> > {
   static CollFuncTable*
   Generate() {
      return CFTGenerator<MapInsert<std::map<K, T, R, A> > >::Generate();
   }


};
// Specialization for std::multimap
template <class K, class T, class R, class A> struct Proxy<std::multimap<K, T, R, A> > {
   static CollFuncTable*
   Generate() {
      return CFTGenerator<MapInsert<std::multimap<K, T, R, A> > >::Generate();
   }


};
// Specialization for std::queue -- not implemented
template <class K, class T> struct Proxy<std::queue<K, T> > {
   static CollFuncTable*
   Generate() { return CFTNullGenerator::Generate(); }

};
// Specialization for std::stack -- not implemented
template <class K, class T> struct Proxy<std::stack<K, T> > {
   static CollFuncTable*
   Generate() { return CFTNullGenerator::Generate(); }

};
#if defined(__GNUC__)
// Specialization for __gnu_cxx::hash_set
template <class T, class F, class E, class A> struct Proxy<__gnu_cxx::hash_set<T, F, E, A> > {
   static CollFuncTable*
   Generate() {
      return CFTGenerator<Insert<__gnu_cxx::hash_set<T, F, E, A> > >::Generate();
   }


};
// Specialization for __gnu_cxx::hash_multiset
template <class T, class F, class E, class A> struct Proxy<__gnu_cxx::hash_multiset<T, F, E, A> > {
   static CollFuncTable*
   Generate() {
      return CFTGenerator<Insert<__gnu_cxx::hash_multiset<T, F, E, A> > >::Generate();
   }


};
// Specialization for __gnu_cxx::hash_map
template <class K, class T, class F, class E, class A> struct Proxy<__gnu_cxx::hash_map<K, T, F, E, A> > {
   static CollFuncTable*
   Generate() {
      return CFTGenerator<MapInsert<__gnu_cxx::hash_map<K, T, F, E, A> > >::Generate();
   }


};
// Specialization for __gnu_cxx::hash_multimap
template <class K, class T, class F, class E, class A> struct Proxy<__gnu_cxx::hash_multimap<K, T, F, E, A> > {
   static CollFuncTable*
   Generate() {
      return CFTGenerator<MapInsert<__gnu_cxx::hash_multimap<K, T, F, E, A> > >::Generate();
   }


};
#elif defined(_WIN32)
// Specialization for stdext::hash_multiset
template <class K, class T, class A> struct Proxy<stdext::hash_multiset<K, T, A> > {
   static CollFuncTable*
   Generate() {
      return CFTGenerator<Insert<stdext::hash_multiset<K, T, A> > >::Generate();
   }


};
// Specialization for stdext::hash_set
template <class K, class T, class A> struct Proxy<stdext::hash_set<K, T, A> > {
   static CollFuncTable*
   Generate() {
      return CFTGenerator<Insert<stdext::hash_set<K, T, A> > >::Generate();
   }


};
// Specialization for stdext::hash_map
template <class K, class T, class R, class A> struct Proxy<stdext::hash_map<K, T, R, A> > {
   static CollFuncTable*
   Generate() {
      return CFTGenerator<MapInsert<stdext::hash_map<K, T, R, A> > >::Generate();
   }


};
// Specialization for stdext::hash_multimap
template <class K, class T, class R, class A> struct Proxy<stdext::hash_multimap<K, T, R, A> > {
   static CollFuncTable*
   Generate() {
      return CFTGenerator<MapInsert<stdext::hash_multimap<K, T, R, A> > >::Generate();
   }


};
#endif

// Specialization for std::bitset
template <typename B> struct StdBitSetHelper {};


#ifndef __CINT__
template <typename Bitset_t> struct CollType<StdBitSetHelper<Bitset_t> > : public Address<const bool&> {
   typedef Bitset_t Cont_t;
   typedef std::pair<size_t, bool> Iter_t;
   typedef bool Value_t;
   typedef Environ<Iter_t> Env_t;
   typedef Env_t* PEnv_t;
   typedef Cont_t* PCont_t;
   typedef Value_t* PValue_t;

   virtual ~CollType() {}

   static inline PCont_t
   object(void* ptr) {
      return PCont_t(PEnv_t(ptr)->fObject);
   }


   static void*
   size(void* env) {
      PEnv_t e = PEnv_t(env);
      e->fSize = PCont_t(e->fObject)->size();
      return &e->fSize;
   }


   static void*
   clear(void* env) {
      object(env)->reset();
      return 0;
   }


   static void*
   first(void* env) {
      PEnv_t e = PEnv_t(env);
      PCont_t c = PCont_t(e->fObject);
      e->fIterator.first = 0;
      e->fIterator.second = c->size() > 0 ? c->test(e->fIterator.first) : false;      // Iterator actually hold the value.
      e->fSize = c->size();
      return 0;
   }


   static void*
   next(void* env) {
      PEnv_t e = PEnv_t(env);
      PCont_t c = PCont_t(e->fObject);

      for ( ; e->fIdx > 0 && e->fIterator.first != c->size(); ++(e->fIterator.first), --e->fIdx) {}
      e->fIterator.second = (e->fIterator.first != c->size()) ? c->test(e->fIterator.first) : false;
      return 0;
   }


   static void*
   construct(void*,size_t) {
      // Nothing to construct.
      return 0;
   }


   static void*
   collect(void* env) {
      PEnv_t e = PEnv_t(env);
      PCont_t c = PCont_t(e->fObject);
      PValue_t m = PValue_t(e->fStart);    // 'start' is a buffer outside the container.

      for (size_t i = 0; i != c->size(); ++i, ++m) {
         *m = c->test(i);
      }
      return 0;
   }


   static void
   destruct(void*,size_t) {
      // Nothing to destruct.
   }


};

template <typename Bitset_t>
struct Pushback<StdBitSetHelper<Bitset_t> > : public CollType<StdBitSetHelper<Bitset_t> > {
   typedef Bitset_t Cont_t;
   typedef bool Iter_t;
   typedef bool Value_t;
   typedef Environ<Iter_t> Env_t;
   typedef Env_t* PEnv_t;
   typedef Cont_t* PCont_t;
   typedef Value_t* PValue_t;

   static void resize(void* /*obj*/, size_t) { }


   static void*
   feed(void* env) {
      PEnv_t e = PEnv_t(env);
      PCont_t c = PCont_t(e->fObject);
      PValue_t m = PValue_t(e->fStart);    // Here start is actually a 'buffer' outside the container.

      for (size_t i = 0; i < e->fSize; ++i, ++m) {
         c->set(i, *m);
      }
      return 0;
   }

   static void*
      feed(void* from, void* to, size_t size) {
      PValue_t m = PValue_t(from);
      PCont_t c  = PCont_t(to);

      for (size_t i = 0; i < size; ++i, ++m) {
         c->set(i, *m);
      }
      return 0;
   }


   static int
   value_offset() {
      return 0;
   }


};
#endif

template <typename B> struct Proxy<StdBitSetHelper<B> > {
   static CollFuncTable*
   Generate() {
      return CFTGenerator<Pushback<StdBitSetHelper<B> > >::Generate();
   }


};
}

#endif // Reflex_CollectionProxy
