// @(#)root/thread:$Id$
/*
 * Copyright (c) 2006-2011 High Performance Computing Center Stuttgart, 
 *                         University of Stuttgart.  All rights reserved.
 * Author: Rainer Keller, HLRS
 * Modified: Fons Rademakers, CERN
 * Modified: Philippe Canal, FNAL
 *
 * Thread-local storage (TLS) is not supported on all environments.
 * This header file and test-program shows how to abstract away, using either
 *   __thread,
 *   __declspec(thread),
 *   thread_local or
 *   Pthread-Keys
 * depending on the (configure-set) CPP-variables R__HAS___THREAD,
 * R__HAS_DECLSPEC_THREAD, R__HAS_THREAD_LOCAL or R__HAS_PTHREAD.
 *
 * Use the macros TTHREAD_TLS_DECLARE, TTHREAD_TLS_INIT, and the
 * getters and setters TTHREAD_TLS_GET and TTHREAD_TLS_GET
 * to work on the declared variables.
 *
 * In case of PThread keys, we need to resolve to using keys!
 * In order to do so, we need to declare and access
 * TLS variables through three macros:
 *  - TTHREAD_TLS_DECLARE
 *  - TTHREAD_TLS_INIT
 *  - TTHREAD_TLS_SET and
 *  - TTHREAD_TLS_GET
 * We do depend on the following (GCC-)extension:
 *  - In case of function-local static functions,
 *    we declare a sub-function to create a specific key.
 * Unfortunately, we do NOT use the following extensions:
 *  - Using typeof, we could get rid of the type-declaration
 *    which is used for casting, however typeof is not ANSI C.
 *  - We do NOT allow something like
 *       func (a, TTHREAD_TLS_SET(int, my_var, 5));
 *    as we do not use the gcc-extension of returning macro-values.
 */
 
#ifndef ROOT_ThreadLocalStorage
#define ROOT_ThreadLocalStorage

#ifndef ROOT_RConfig
#include "RConfig.h"
#endif

#ifndef ROOT_RConfigure
#include "RConfigure.h"
#endif

#if defined(R__MACOSX)
#  if defined(__clang__) && defined(MAC_OS_X_VERSION_10_7) && (defined(__x86_64__) || defined(__i386__))
#    define R__HAS___THREAD
#  elif !defined(R__HAS_PTHREAD)
#    define R__HAS_PTHREAD
#  endif
#endif
#if defined(R__LINUX) || defined(R__AIX)
#  define R__HAS___THREAD
#endif
#if defined(R__SOLARIS) && !defined(R__HAS_PTHREAD)
#  define R__HAS_PTHREAD
#endif
#if defined(R__WIN32)
#  define R__HAS_DECLSPEC_THREAD
#endif

#if __cplusplus >= 201103L

// Note: it would be tempting to use __has_feature(cxx_thread_local) but despite
// documentation claims it support for it ... it is in fact ineffective (return
// a false negative).
// Clang 3.4 also support SD-6 (feature test macros __cpp_*), but no thread local macro
#  if defined(__clang__) && (__clang_major__ >= 3 && __clang_minor__ >= 3)
     // thread_local was added in Clang 3.3
     // Still requires libstdc++ from GCC 4.8
     // For that __GLIBCXX__ isn't good enough
#    define R__HAS_THREAD_LOCAL

#  elif defined(__GNUG__) && (__GNUC__ <= 4 && __GNUC_MINOR__ < 80)
    // The C++11 thread_local keyword is supported in GCC only since 4.8
#    define R__HAS___THREAD
#  else
#    define R__HAS_THREAD_LOCAL
#  endif

#endif


#ifdef __cplusplus

// Note that the order is relevant, more than one of the flag might be
// on at the same time and we want to use 'best' option available.

#ifdef __CINT__

#  define TTHREAD_TLS(type) static type
#  define TTHREAD_TLS_ARRAY(type,size,name) static type name[size];
#  define TTHREAD_TLS_PTR(name) &name

#elif defined(R__HAS_THREAD_LOCAL)

#  define TTHREAD_TLS(type) thread_local type
#  define TTHREAD_TLS_ARRAY(type,size,name) thread_local type name[size];
#  define TTHREAD_TLS_PTR(name) &name

#elif defined(R__HAS___THREAD)

#  define TTHREAD_TLS(type)  static __thread type
#  define TTHREAD_TLS_ARRAY(type,size,name) static __thread type name[size];
#  define TTHREAD_TLS_PTR(name) &name

#elif defined(R__HAS_DECLSPEC_THREAD)

#  define TTHREAD_TLS(type) static __declspec(thread) type
#  define TTHREAD_TLS_ARRAY(type,size,name) static __declspec(thread) type name[size];
#  define TTHREAD_TLS_PTR(name) &name

#elif defined(R__HAS_PTHREAD)

#include <assert.h>
#include <pthread.h>

template <typename type> class TThreadTLSWrapper
{
private:
   pthread_key_t  fKey;
   type           fInitValue;
   
   static void key_delete(void *arg) {
      assert (NULL != arg);
      delete (type*)(arg);
   }

public:

   TThreadTLSWrapper() : fInitValue() {

      pthread_key_create(&(fKey), key_delete);
   }

   TThreadTLSWrapper(const type &value) : fInitValue(value) {

      pthread_key_create(&(fKey), key_delete);
   }

   ~TThreadTLSWrapper() {
      pthread_key_delete(fKey);
   }

   type& get() {
      void *ptr = pthread_getspecific(fKey);
      if (!ptr) {
         ptr = new type(fInitValue);
         assert (NULL != ptr);
         (void) pthread_setspecific(fKey, ptr);
      }
      return *(type*)ptr;
   }

   type& operator=(const type &in) {
      type &ptr = get();
      ptr = in;
      return ptr;
   }

   operator type&() {
      return get();
   }

};

template <typename type,int size> class TThreadTLSArrayWrapper
{
private:
   pthread_key_t  fKey;

   static void key_delete(void *arg) {
      assert (NULL != arg);
      delete [] (type*)(arg);
   }

public:

   TThreadTLSArrayWrapper() {

      pthread_key_create(&(fKey), key_delete);
   }

   ~TThreadTLSArrayWrapper() {
      pthread_key_delete(fKey);
   }

   type* get() {
      void *ptr = pthread_getspecific(fKey);
      if (!ptr) {
         ptr = new type[size];
         assert (NULL != ptr);
         (void) pthread_setspecific(fKey, ptr);
      }
      return  (type*)ptr;
   }

//   type& operator=(const type &in) {
//      type &ptr = get();
//      ptr = in;
//      return ptr;
//   }
//
   operator type*() {
      return get();
   }
   
};

#  define TTHREAD_TLS(type) static TThreadTLSWrapper<type>
#  define TTHREAD_TLS_ARRAY(type,size,name) static TThreadTLSArrayWrapper<type,size> name;
#  define TTHREAD_TLS_PTR(name) &(name.get())
#else

#error "No Thread Local Storage (TLS) technology for this platform specified."

#endif

// Available on all platforms
template <int marker, typename T>
T &TTHREAD_TLS_INIT() {
   TTHREAD_TLS(T*) ptr = NULL;
   TTHREAD_TLS(Bool_t) isInit(kFALSE);
   if (!isInit) {
      ptr = new T;
      isInit = kTRUE;
   }
   return *ptr;
}

template <int marker, typename Array, typename T>
Array &TTHREAD_TLS_INIT_ARRAY() {
   TTHREAD_TLS(Array*) ptr = NULL;
   TTHREAD_TLS(Bool_t) isInit(kFALSE);
   if (!isInit) {
      ptr = new Array[sizeof(Array)/sizeof(T)];
      isInit = kTRUE;
   }
   return *ptr;
}

template <int marker, typename T, typename ArgType>
T &TTHREAD_TLS_INIT(ArgType arg) {
   TTHREAD_TLS(T*) ptr = NULL;
   TTHREAD_TLS(Bool_t) isInit(kFALSE);
   if (!isInit) {
      ptr = new T(arg);
      isInit = kTRUE;
   }
   return *ptr;
}

#else // __cplusplus

#if defined(R__HAS_THREAD_LOCAL)

#  define TTHREAD_TLS_DECLARE(type,name)
#  define TTHREAD_TLS_INIT(type,name,value) thread_local type name = (value)
#  define TTHREAD_TLS_SET(type,name,value)  name = (value)
#  define TTHREAD_TLS_GET(type,name)        (name)
#  define TTHREAD_TLS_FREE(name)

#elif defined(R__HAS___THREAD)

#  define TTHREAD_TLS_DECLARE(type,name)
#  define TTHREAD_TLS_INIT(type,name,value) static __thread type name = (value)
#  define TTHREAD_TLS_SET(type,name,value)  name = (value)
#  define TTHREAD_TLS_GET(type,name)        (name)
#  define TTHREAD_TLS_FREE(name)

#elif defined(R__HAS_DECLSPEC_THREAD)

#  define TTHREAD_TLS_DECLARE(type,name)
#  define TTHREAD_TLS_INIT(type,name,value) static __declspec(thread) type name = (value)
#  define TTHREAD_TLS_SET(type,name,value)  name = (value)
#  define TTHREAD_TLS_GET(type,name)        (name)
#  define TTHREAD_TLS_FREE(name)

#elif defined(R__HAS_PTHREAD)

#include <assert.h>
#include <pthread.h>

#  define TTHREAD_TLS_DECLARE(type,name)                                     \
   static pthread_key_t _##name##_key;                                       \
   static void _##name##_key_delete(void * arg)                              \
   {                                                                         \
     assert (NULL != arg);                                                   \
     free(arg);                                                              \
   }                                                                         \
   static void _##name##_key_create(void)                                    \
   {                                                                         \
     int _ret;                                                               \
     _ret = pthread_key_create(&(_##name##_key), _##name##_key_delete);      \
     _ret = _ret; /* To get rid of warnings in case of NDEBUG */             \
     assert (0 == _ret);                                                     \
   }                                                                         \
   static pthread_once_t _##name##_once = PTHREAD_ONCE_INIT;

#  define TTHREAD_TLS_INIT(type,name,value)                                  \
   do {                                                                      \
     void *_ptr;                                                             \
     (void) pthread_once(&(_##name##_once), _##name##_key_create);           \
     if ((_ptr = pthread_getspecific(_##name##_key)) == NULL) {              \
       _ptr = malloc(sizeof(type));                                          \
       assert (NULL != _ptr);                                                \
       (void) pthread_setspecific(_##name##_key, _ptr);                      \
       *((type*)_ptr) = (value);                                             \
     }                                                                       \
   } while (0)

#  define TTHREAD_TLS_SET(type,name,value)                                   \
   do {                                                                      \
     void *_ptr = pthread_getspecific(_##name##_key);                        \
     assert (NULL != _ptr);                                                  \
     *((type*)_ptr) = (value);                                               \
   } while (0)

#  define TTHREAD_TLS_GET(type,name)                                         \
     *((type*)pthread_getspecific(_##name##_key))

#  define TTHREAD_TLS_FREE(name)                                             \
     pthread_key_delete(_##name##_key);

#else

#error "No Thread Local Storage (TLS) technology for this platform specified."

#endif

#endif // __cplusplus

#endif

