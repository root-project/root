// @(#)root/cont:$Name:  $:$Id: TCollectionProxy.h,v 1.2 2004/11/01 12:26:07 brun Exp $
// Author: Markus Frank  28/10/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TCollectionProxy
#define ROOT_TCollectionProxy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  Small helper to save proxy environment in the event of
//  recursive calls.
//
//////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include <typeinfo>

// Forward declarations
class TBuffer;
class TClassStreamer;
class TMemberStreamer;
class TVirtualCollectionProxy;

#if defined(_WIN32) 
  #if _MSC_VER<1300
    #define TYPENAME 
  #else
    #define TYPENAME typename
  #endif
#else
  #define TYPENAME typename
#endif

namespace ROOT {
  /** @class TCollectionProxy::Environ TCollectionProxy.h TCollectionProxy.h 
    *
    * Small helper to save proxy environment in the event of
    * recursive calls.
    *
    * @author  M.Frank
    * @version 1.0
    * @date    10/10/2004
    */
  template <class T> struct Environ  {
    typedef T           Iter_t;
    char                buff[64];
    size_t              idx;
    size_t              size;
    void*               object;
    void*               start;
    void*               temp;
    bool                delete_temp;
    int                 refCount;
    T& iter() { return *(T*)buff; }
  };
}

/** @class TCollectionProxy TCollectionProxy.h cont/TCollectionProxy.h
  *
  * TCollectionProxy
  * Interface to collection proxy and streamer generator.
  *
  * Proxy around an arbitrary container, which implements basic 
  * functionality and iteration. The purpose of this implementation 
  * is to shield any generated dictionary implementation from the
  * underlying streamer/proxy implementation and only expose
  * the creation fucntions.
  *
  * In particular this is used to implement splitting and abstract
  * element access of any container. Access to compiled code is necessary
  * to implement the abstract iteration sequence and functionality like
  * size(), clear(), resize(). resize() may be a void operation.
  *
  * @author  M.Frank
  * @version 1.0
  */
class TCollectionProxy  {
public:

  typedef TVirtualCollectionProxy Proxy_t;
#ifdef R__HPUX
  typedef const type_info&      Info_t;
#else
  typedef const std::type_info& Info_t;
#endif
  template <class T> struct Address {
    static void* address(T ref) {
      return (void*)&ref;
    }
  };

  /** @class TCollectionProxy::fType TCollectionProxy.h TCollectionProxy.h 
    *
    * Small helper to encapsulate basic data accesses for 
    * all STL continers.
    *
    * @author  M.Frank
    * @version 1.0
    * @date    10/10/2004
    */
  template <class T> struct Type : public Address<TYPENAME T::const_reference> {
    typedef T                      Cont_t;
    typedef typename T::iterator   Iter_t;
    typedef typename T::value_type Value_t;
    typedef ROOT::Environ<Iter_t>  Env_t;
    typedef Env_t                 *PEnv_t;
    typedef Cont_t                *PCont_t;
    typedef Value_t               *PValue_t;

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
      TYPENAME T::reference ref = *(e->iter());
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
      TYPENAME T::reference ref = *(e->iter());
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
      PCont_t  c = PCont_t(e->object);
      PValue_t m = PValue_t(e->start);
      TYPENAME T::allocator_type a = c->get_allocator();
      for (size_t i=0; i < e->size; ++i, ++m )
        a.destroy(m);
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
  template <class T> struct Pushback : public Type<T> {
    typedef T                      Cont_t;
    typedef typename T::iterator   Iter_t;
    typedef typename T::value_type Value_t;
    typedef ROOT::Environ<Iter_t>  Env_t;
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
  template <class T> struct Insert : public Type<T> {
    typedef T                      Cont_t;
    typedef typename T::iterator   Iter_t;
    typedef typename T::value_type Value_t;
    typedef ROOT::Environ<Iter_t>  Env_t;
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
    typedef ROOT::Environ<Iter_t>  Env_t;
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

  /// Generate emulated collection proxy for a given class
  static Proxy_t* genEmulatedProxy(const char* class_name);

  /// Generate emulated class streamer for a given collection class
  static TClassStreamer* genEmulatedClassStreamer(const char* class_name);

  /// Generate emulated member streamer for a given collection class
  static TMemberStreamer* genEmulatedMemberStreamer(const char* class_name);

  /// Generate proxy from static functions
  static Proxy_t* genExplicitProxy( Info_t info,
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
                                    );

  /// Generate proxy from template
  template <class T> static Proxy_t* genProxy(const T&)  {
    std::pair<TYPENAME T::Value_t, TYPENAME T::Value_t>* p = 
       (std::pair<TYPENAME T::Value_t, TYPENAME T::Value_t>*)0x1000;
    return genExplicitProxy(typeid(TYPENAME T::Cont_t),
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

  /// Generate streamer from static functions
  static Proxy_t* 
    genExplicitStreamer(  Info_t  info,
                          size_t  iter_size,
                          size_t  value_diff,
                          int     value_offset,
                          void*  (*size_func)(void*),
                          void*  (*resize_func)(void*),
                          void*  (*clear_func)(void*),
                          void*  (*first_func)(void*),
                          void*  (*next_func)(void*),
                          void*  (*construct_func)(void*),
                          void*  (*destruct_func)(void*),
                          void*  (*feed_func)(void*),
                          void*  (*collect_func)(void*)
                          );

  /// Generate class streamer from static functions
  static TClassStreamer* 
    genExplicitClassStreamer( Info_t  info,
                              size_t  iter_size,
                              size_t  value_diff,
                              int     value_offset,
                              void*  (*size_func)(void*),
                              void*  (*resize_func)(void*),
                              void*  (*clear_func)(void*),
                              void*  (*first_func)(void*),
                              void*  (*next_func)(void*),
                              void*  (*construct_func)(void*),
                              void*  (*destruct_func)(void*),
                              void*  (*feed_func)(void*),
                              void*  (*collect_func)(void*)
                              );

  /// Generate class streamer from template
  template <class T> static TClassStreamer* genClassStreamer(const T&)  {
    std::pair<TYPENAME T::Value_t, TYPENAME T::Value_t>* p = 
       (std::pair<TYPENAME T::Value_t, TYPENAME T::Value_t>*)0x1000;
    return genExplicitClassStreamer(typeid(TYPENAME T::Cont_t),
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

  /// Generate member streamer from static functions
  static TMemberStreamer* 
    genExplicitMemberStreamer(Info_t  info,
                              size_t  iter_size,
                              size_t  value_diff,
                              int     value_offset,
                              void*  (*size_func)(void*),
                              void*  (*resize_func)(void*),
                              void*  (*clear_func)(void*),
                              void*  (*first_func)(void*),
                              void*  (*next_func)(void*),
                              void*  (*construct_func)(void*),
                              void*  (*destruct_func)(void*),
                              void*  (*feed_func)(void*),
                              void*  (*collect_func)(void*)
                              );

  /// Generate member streamer from template
  template <class T> static TMemberStreamer* genMemberStreamer(const T&)  {
    std::pair<TYPENAME T::Value_t, TYPENAME T::Value_t>* p = 
       (std::pair<TYPENAME T::Value_t, TYPENAME T::Value_t>*)0x1000;
    return genExplicitMemberStreamer( typeid(TYPENAME T::Cont_t),
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

/** @class TCollectionStreamer TEmulatedCollectionProxy.h cont/TEmulatedCollectionProxy.h
  *
  * TEmulatedClassStreamer
  *
  * Class streamer object to implement TClassStreamr functionality
  * for I/O emulation.
  *
  * @author  M.Frank
  * @version 1.0
  */
class TCollectionStreamer   {
protected:
  TVirtualCollectionProxy* fProxy;   /// Pointer to worker proxy

  /// Issue Error about invalid proxy
  void InvalidProxyError();

public:
  /// Initializing constructor
  TCollectionStreamer();
  /// Copy constructor
  TCollectionStreamer(const TCollectionStreamer& c);
  /// Standard destructor
  virtual ~TCollectionStreamer();
  /// Attach worker proxy
  void AdoptProxy(TVirtualCollectionProxy* proxy);
  /// Streamer for I/O handling
  void Streamer(TBuffer &refBuffer, void *pObject, int siz);
};

#include "TClassStreamer.h"

/** @class TEmulatedClassStreamer TEmulatedCollectionProxy.h cont/TEmulatedCollectionProxy.h
  *
  * TEmulatedClassStreamer
  *
  * Class streamer object to implement TClassStreamr functionality
  * for I/O emulation.
  *
  * @author  M.Frank
  * @version 1.0
  */
class TCollectionClassStreamer : public TClassStreamer, public TCollectionStreamer {
public:
  /// Initializing constructor
  TCollectionClassStreamer() : TClassStreamer(0)     {                        }
  /// Copy constructor
  TCollectionClassStreamer(const TCollectionClassStreamer& c) 
    : TClassStreamer(c), TCollectionStreamer(c)      {                        }
  /// Standard destructor
  virtual ~TCollectionClassStreamer()                {                        }
  /// Streamer for I/O handling
  virtual void operator()(TBuffer &buff, void *pObj) { Streamer(buff,pObj,0); }
};

#include "TMemberStreamer.h"

/** @class TCollectionMemberStreamer TEmulatedCollectionProxy.h cont/TEmulatedCollectionProxy.h
  *
  * TCollectionMemberStreamer
  *
  * Class streamer object to implement TMemberStreamer functionality
  * for I/O emulation.
  *
  * @author  M.Frank
  * @version 1.0
  */
class TCollectionMemberStreamer : public TMemberStreamer, public TCollectionStreamer {
public:
  /// Initializing constructor
  TCollectionMemberStreamer() : TMemberStreamer(0) { }
  /// Copy constructor
  TCollectionMemberStreamer(const TCollectionMemberStreamer& c) 
    : TMemberStreamer(c), TCollectionStreamer(c)   { }
  /// Standard destructor
  virtual ~TCollectionMemberStreamer()             { }
  /// Streamer for I/O handling
  virtual void operator()(TBuffer &buff,void *pObj,Int_t siz=0)
  { Streamer(buff, pObj, siz);                       }
};

// Need specialization for boolean references due to stupid STL vector<bool>
template<> inline void* TCollectionProxy::Address<std::vector<bool>::const_reference>::address(std::vector<bool>::const_reference ) {
  return 0;
}
#endif
#endif
