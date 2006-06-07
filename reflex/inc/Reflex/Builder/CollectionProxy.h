// @(#)root/reflex:$Name:  $:$Id: CollectionProxy.h,v 1.13 2006/06/06 16:07:15 roiser Exp $
// Author: Markus Frank 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_CollectionProxy
#define ROOT_Reflex_CollectionProxy 1 1

#include <cstddef>

// Forward declarations
namespace std {
   template <class T, class A>  class deque;
   template <class T, class A>  class vector;
   template <class T, class A>  class list;
   template <class T, class A>  class queue;
   template <class T, class A>  class stack;
   template <class K, class T, class A>  class set;
   template <class K, class T, class A>  class multiset;
   template <class K, class T, class R, class A>  class map;
   template <class K, class T, class R, class A>  class multimap;
   template <class T> class allocator;
}
// Hash map forward declarations
#if defined(__GNUC__)
namespace __gnu_cxx {  // GNU GCC
   template <class T, class F, class E, class A>  class hash_set;
   template <class T, class F, class E, class A>  class hash_multiset;
   template <class K, class T, class F, class E, class A>  class hash_map;
   template <class K, class T, class F, class E, class A>  class hash_multimap;
}
#elif  defined(_WIN32)
namespace stdext {     // Visual C++
   template <class K, class T, class A>  class hash_set;
   template <class K, class T, class A>  class hash_multiset;
   template <class K, class T, class R, class A>  class hash_map;
   template <class K, class T, class R, class A>  class hash_multimap;
}
#endif

namespace ROOT {
   namespace Reflex  {

#ifndef __CINT__
      template <typename T> struct Environ  {
         typedef T           Iter_t;
         char                buff[64];
         size_t              idx;
         size_t              size;
         void*               object;
         void*               start;
         void*               temp;
         bool                delete_temp;
         int                 refSize;
         size_t              space;
         T& iter() { return *(T*)buff; }
      };
#else 
      template <typename T> struct Environ;
#endif

      template <typename T> struct Address {
         static void* address(T ref) {
            return (void*)&ref;
         }
      };

      template <class T> struct CollType 
#ifdef _KCC  // KAI compiler
         : public Address<typename T::value_type&> 
#else 
         : public Address<typename T::const_reference> 
#endif 
      {
         typedef T                               Cont_t;
         typedef typename T::iterator            Iter_t;
         typedef typename T::value_type          Value_t;
         typedef ROOT::Reflex::Environ<Iter_t>   Env_t;
         typedef Env_t                          *PEnv_t;
         typedef Cont_t                         *PCont_t;
         typedef Value_t                        *PValue_t;

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
            e->idx = 0;
            PCont_t c = PCont_t(e->object);
            // Assume iterators do not need destruction
            ::new(e->buff) Iter_t(c->begin()); 
            e->size  = c->size();
            if ( 0 == e->size ) return e->start = 0;
#ifdef _KCC  // KAI compiler
            typename T::value_type& ref = *(e->iter());
#else
            typename T::const_reference ref = *(e->iter());
#endif
            return e->start = address(ref);
         }
         static void* next(void* env)  {
            PEnv_t  e = PEnv_t(env);
            ++e->idx;
            PCont_t c = PCont_t(e->object);
            for (; e->idx > 0 && e->iter() != c->end(); ++(e->iter()), --e->idx );
            // TODO: Need to find something for going backwards....
            if ( e->iter() == c->end() ) return 0;
#ifdef _KCC  // KAI compiler
            typename T::value_type& ref = *(e->iter());
#else
            typename T::const_reference ref = *(e->iter());
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
            for (size_t i=0; i < e->size; ++i, ++m )
               m->~Value_t();
            return 0;
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
      template <class T> struct Pushback : public CollType<T> {
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

      /** @class TCollectionProxy::Map TCollectionProxy.h TCollectionProxy.h 
       *
       * Small helper to encapsulate all necessary data accesses for 
       * containers like set, multiset etc.
       *
       * @author  M.Frank
       * @version 1.0
       * @date    10/10/2004
       */
      template <class T> struct Insert : public CollType<T> {
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

      /** @class TCollectionProxy::Map TCollectionProxy.h TCollectionProxy.h 
       *
       * Small helper to encapsulate all necessary data accesses for 
       * containers like Set, multiset etc.
       *
       * @author  M.Frank
       * @version 1.0
       * @date    10/10/2004
       */
      template <class T> struct MapInsert : public CollType<T> {
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

      // Need specialization for boolean references due to stupid STL vector<bool>
      template<> inline void* ROOT::Reflex::Address<std::vector<bool,std::allocator<bool> >::const_reference>::address(std::vector<bool,std::allocator<bool> >::const_reference ) {
         return 0;
      }

   }
}

#include <vector>
namespace ROOT {
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
         int    value_offset;
         void*  (*size_func)(void*);
         void*  (*resize_func)(void*);
         void*  (*clear_func)(void*);
         void*  (*first_func)(void*);
         void*  (*next_func)(void*);
         void*  (*construct_func)(void*);
         void*  (*destruct_func)(void*);
         void*  (*feed_func)(void*);
         void*  (*collect_func)(void*);
      };

      template <typename T> struct CFTGenerator {
         static  CollFuncTable* Generate()  {
            typedef typename T::Value_t Value_t;
            typedef std::pair<Value_t,Value_t> Pair_t;
            Pair_t* ptr = (Pair_t*)0x1000;
            CollFuncTable*  p  = new CollFuncTable();
            p->iter_size       = sizeof(typename T::Iter_t);
            p->value_diff      = ((char*)&ptr->second) - ((char*)&ptr->first);
            p->value_offset    = T::value_offset();
            p->size_func       = T::size;
            p->first_func      = T::first;
            p->next_func       = T::next;
            p->clear_func      = T::clear;
            p->resize_func     = T::resize;
            p->collect_func    = T::collect;
            p->construct_func  = T::construct;
            p->destruct_func   = T::destruct;
            p->feed_func       = T::feed;
            return p;
         }
      };
      struct CFTNullGenerator {
         static void* Void_func(void*) {
            return 0;
         }
         static  CollFuncTable* Generate()  {
            CollFuncTable*  p  = new CollFuncTable();
            p->iter_size       = 4;
            p->value_diff      = 0;
            p->value_offset    = 0;
            p->size_func       = Void_func;
            p->first_func      = Void_func;
            p->next_func       = Void_func;
            p->clear_func      = Void_func;
            p->resize_func     = Void_func;
            p->collect_func    = Void_func;
            p->construct_func  = Void_func;
            p->destruct_func   = Void_func;
            p->feed_func       = Void_func;
            return p;
         }
      };
      // General proxy (dummy)
      template <typename A> struct Proxy {};

      // Specialization for std::vector 
      template <class T, class A> struct Proxy< std::vector<T,A> > {
         static CollFuncTable* Generate()  {
            return CFTGenerator<Pushback<std::vector<T,A> > >::Generate();
         }
      };
      // Specialization for std::list 
      template <class T, class A> struct Proxy< std::list<T,A> > {
         static CollFuncTable* Generate()  {
            return CFTGenerator<Pushback<std::list<T,A> > >::Generate();
         }
      };
      // Specialization for std::deque 
      template <class T, class A> struct Proxy< std::deque<T,A> > {
         static CollFuncTable* Generate()  {
            return CFTGenerator<Pushback<std::deque<T,A> > >::Generate();
         }
      };
      // Specialization for std::set 
      template <class K, class T, class A> struct Proxy< std::set<K,T,A> > {
         static CollFuncTable* Generate()  {
            return CFTGenerator<Insert<std::set<K,T,A> > >::Generate();
         }
      };
      // Specialization for std::multiset 
      template <class K, class T, class A> struct Proxy< std::multiset<K,T,A> > {
         static CollFuncTable* Generate()  {
            return CFTGenerator<Insert<std::multiset<K,T,A> > >::Generate();
         }
      };
      // Specialization for std::map 
      template <class K, class T, class R, class A> struct Proxy< std::map<K,T,R,A> > {
         static CollFuncTable* Generate()  {
            return CFTGenerator<MapInsert<std::map<K,T,R,A> > >::Generate();
         }
      };
      // Specialization for std::multimap 
      template <class K, class T, class R, class A> struct Proxy< std::multimap<K,T,R,A> > {
         static CollFuncTable* Generate()  {
            return CFTGenerator<MapInsert<std::multimap<K,T,R,A> > >::Generate();
         }
      };
      // Specialization for std::queue -- not implemented 
      template <class K, class T> struct Proxy< std::queue<K,T> > {
         static CollFuncTable* Generate()  { return CFTNullGenerator::Generate(); }
      };
      // Specialization for std::stack -- not implemented 
      template <class K, class T> struct Proxy< std::stack<K,T> > {
         static CollFuncTable* Generate()  { return CFTNullGenerator::Generate(); }
      };
#if defined(__GNUC__)
      // Specialization for __gnu_cxx::hash_set 
      template <class T, class F, class E, class A> struct Proxy< __gnu_cxx::hash_set<T,F,E,A> > {
         static CollFuncTable* Generate()  {
            return CFTGenerator<Insert<__gnu_cxx::hash_set<T,F,E,A> > >::Generate();
         }
      };
      // Specialization for __gnu_cxx::hash_multiset 
      template <class T, class F, class E, class A> struct Proxy< __gnu_cxx::hash_multiset<T,F,E,A> > {
         static CollFuncTable* Generate()  {
            return CFTGenerator<Insert<__gnu_cxx::hash_multiset<T,F,E,A> > >::Generate();
         }
      };
      // Specialization for __gnu_cxx::hash_map 
      template <class K, class T, class F, class E, class A> struct Proxy< __gnu_cxx::hash_map<K,T,F,E,A> > {
         static CollFuncTable* Generate()  {
            return CFTGenerator<MapInsert<__gnu_cxx::hash_map<K,T,F,E,A> > >::Generate();
         }
      };
      // Specialization for __gnu_cxx::hash_multimap 
      template <class K, class T, class F, class E, class A> struct Proxy< __gnu_cxx::hash_multimap<K,T,F,E,A> > {
         static CollFuncTable* Generate()  {
            return CFTGenerator<MapInsert<__gnu_cxx::hash_multimap<K,T,F,E,A> > >::Generate();
         }
      };
#elif defined(_WIN32)
      // Specialization for stdext::hash_multiset 
      template <class K, class T, class A> struct Proxy< stdext::hash_multiset<K,T,A> > {
         static CollFuncTable* Generate()  {
            return CFTGenerator<Insert<stdext::hash_multiset<K,T,A> > >::Generate();
         }
      };
      // Specialization for stdext::hash_set 
      template <class K, class T, class A> struct Proxy< stdext::hash_set<K,T,A> > {
         static CollFuncTable* Generate()  {
            return CFTGenerator<Insert<stdext::hash_set<K,T,A> > >::Generate();
         }
      };
      // Specialization for stdext::hash_map 
      template <class K, class T, class R, class A> struct Proxy< stdext::hash_map<K,T,R,A> > {
         static CollFuncTable* Generate()  {
            return CFTGenerator<MapInsert<stdext::hash_map<K,T,R,A> > >::Generate();
         }
      };
      // Specialization for stdext::hash_multimap 
      template <class K, class T, class R, class A> struct Proxy< stdext::hash_multimap<K,T,R,A> > {
         static CollFuncTable* Generate()  {
            return CFTGenerator<MapInsert<stdext::hash_multimap<K,T,R,A> > >::Generate();
         }
      };
#endif
   }
}
#endif // ROOT_Reflex_CollectionProxy
