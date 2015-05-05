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
#include <assert.h>
#include <vector>
#include <map>
#include <set>

// Macro indicating the version of the Collection Proxy interface followed
// by this Reflex build, this must match the version number of
// ROOT_COLLECTIONPROXY_VERSION in ROOT's TVirtutalCollectionProxy.h

#define REFLEX_COLLECTIONPROXY_VERSION 3

// Forward declarations

class TVirtualCollectionProxy;

namespace std {
   template <class T, class A> class deque;
   //template <class T, class A> class vector;
   template <class T, class A> class list;
   template <class T, class A> class queue;
   template <class T, class A> class stack;
   //template <class K, class T, class A> class set;
   //template <class K, class T, class A> class multiset;
   //template <class K, class T, class R, class A> class map;
   //template <class K, class T, class R, class A> class multimap;
   //template <class T> class allocator;
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
   
   // Same value as TVirtualCollectionProxy.
   static const unsigned int fgIteratorArenaSize = 16; // greater than sizeof(void*) + sizeof(UInt_t)

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

      template <> struct IteratorValue<std::vector<bool>,bool> {
         static void* get(std::vector<bool>::iterator & /* iter */) {
            return 0;
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

         static void create(void *coll, void **begin_arena, void **end_arena, TVirtualCollectionProxy *) {
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

         static void create(void *coll, void **begin_arena, void **end_arena, TVirtualCollectionProxy *) {
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
            *(void**)dest = *(void**)source;
            return dest; 
         }
         static void* next(void * /* iter_loc */, const void * /* end_loc */) {
            // Should not be used.
            assert(0 && "Intentionally not implemented, do not use.");
            return 0;
         }
         static void destruct1(void  * /* iter_ptr */) {
            // Nothing to do
         }
         static void destruct2(void * /* begin_ptr */, void * /* end_ptr */) {
            // Nothing to do
         }
      };

      template <> struct Iterators<std::vector<bool>, false> {
         typedef std::vector<bool> Cont_t;
         typedef Cont_t *PCont_t;
         typedef Cont_t::iterator iterator;

         static void create(void *coll, void **begin_arena, void **end_arena, TVirtualCollectionProxy *) {
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
               void *result = IteratorValue<Cont_t, Cont_t::value_type>::get(*iter);
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

      template <typename Cont_t> struct Iterators<Cont_t, /* large= */ true > {
         typedef Cont_t *PCont_t;
         typedef typename Cont_t::iterator iterator;

         static void create(void *coll, void **begin_arena, void **end_arena, TVirtualCollectionProxy *) {
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
            collect(void* coll, void *array) {
            PCont_t c = PCont_t(coll);
            PValue_t m = PValue_t(array);

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

         static const bool fgLargeIterator = sizeof(typename Cont_t::iterator) > fgIteratorArenaSize;
         typedef Iterators<Cont_t,fgLargeIterator> Iterators_t;

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
      void*  (*collect_func)(void*,void*);
      void*  (*create_env)();

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

         p->fCreateIterators = T::Iterators_t::create;
         p->fCopyIterator = T::Iterators_t::copy;
         p->fNext = T::Iterators_t::next;
         p->fDeleteSingleIterator = T::Iterators_t::destruct1;
         p->fDeleteTwoIterators = T::Iterators_t::destruct2;
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
      Void_func2c(void*,void*) { return 0; }

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
         p->collect_func = Void_func2c;
         p->construct_func = Void_func2;
         p->destruct_func = Void_func2b;
         p->feed_func = Void_func3;
         p->create_env = Void_func0;

         p->fCreateIterators = 0;
         p->fCopyIterator = 0;
         p->fNext = 0;
         p->fDeleteSingleIterator = 0;
         p->fDeleteTwoIterators = 0;

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
         collect(void* coll, void *array) {
         PCont_t c = PCont_t(coll);
         PValue_t m = PValue_t(array);    // 'start' is a buffer outside the container.

         for (size_t i = 0; i != c->size(); ++i, ++m) {
            *m = c->test(i);
         }
         return 0;
      }


      static void
         destruct(void*,size_t) {
         // Nothing to destruct.
      }

      struct Iterators {
         typedef Cont_t *PCont_t;
         union PtrSize_t { size_t fIndex; void *fAddress; };
         typedef std::pair<PtrSize_t,bool> iterator;
         // In the end iterator we store the bitset pointer
         // and do not use the 'second' part of the pair.
         // In the other iterator we store the index
         // and the value.

         static void create(void *coll, void **begin_arena, void **end_arena, TVirtualCollectionProxy *) {
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
