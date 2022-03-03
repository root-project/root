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
 * R__HAS_DECLSPEC_THREAD or R__HAS_PTHREAD.
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
 *
 * C++11 requires the implementation of the thread_local storage.
 *
 *  For simple type use:
 *      TTHREAD_TLS(int) varname;
 *
 *  For array of simple type use:
 *      TTHREAD_TLS_ARRAY(int, arraysize, varname);
 *
 *  For object use:
 *      TTHREAD_TLS_DECL(classname, varname);
 *      TTHREAD_TLS_DECL_ARG(classname, varname, arg);
 *      TTHREAD_TLS_DECL_ARG2(classname, varname, arg1, arg2);
 *
 */

#ifndef ROOT_ThreadLocalStorage
#define ROOT_ThreadLocalStorage

#include <stddef.h>

#ifdef __cplusplus
#include "RtypesCore.h"
#endif

#include <ROOT/RConfig.hxx>

#include "RConfigure.h"

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

#ifdef __cplusplus

// Note that the order is relevant, more than one of the flag might be
// on at the same time and we want to use 'best' option available.

#ifdef __CINT__

#  define TTHREAD_TLS(type) static type
#  define TTHREAD_TLS_ARRAY(type,size,name) static type name[size]
#  define TTHREAD_TLS_PTR(name) &name

#else

#  define TTHREAD_TLS(type) thread_local type
#  define TTHREAD_TLS_ARRAY(type,size,name) thread_local type name[size]
#  define TTHREAD_TLS_PTR(name) &name

#  define TTHREAD_TLS_DECL(type, name) thread_local type name
#  define TTHREAD_TLS_DECL_ARG(type, name, arg) thread_local type name(arg)
#  define TTHREAD_TLS_DECL_ARG2(type, name, arg1, arg2) thread_local type name(arg1,arg2)

#endif

// Available on all platforms


// Neither TTHREAD_TLS_DECL_IMPL and TTHREAD_TLS_INIT
// do not delete the object at the end of the process.

#define TTHREAD_TLS_DECL_IMPL(type, name, ptr, arg) \
   TTHREAD_TLS(type *) ptr = 0; \
   if (!ptr) ptr = new type(arg); \
   type &name = *ptr;

#define TTHREAD_TLS_DECL_IMPL2(type, name, ptr, arg1, arg2) \
   TTHREAD_TLS(type *) ptr = 0; \
   if (!ptr) ptr = new type(arg1,arg2); \
   type &name = *ptr;

#ifndef TTHREAD_TLS_DECL

#define TTHREAD_TLS_DECL(type, name) \
   TTHREAD_TLS_DECL_IMPL(type,name,_R__JOIN_(ptr,__LINE__),)

#define TTHREAD_TLS_DECL_ARG(type, name, arg) \
   TTHREAD_TLS_DECL_IMPL(type,name,_R__JOIN_(ptr,__LINE__),arg)

#define TTHREAD_TLS_DECL_ARG2(type, name, arg1, arg2) \
   TTHREAD_TLS_DECL_IMPL2(type,name,_R__JOIN_(ptr,__LINE__),arg1,arg2)

#endif

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

// TODO: Evaluate using thread_local / _Thread_local from C11 instead of
// platform-specific solutions such as __thread, __declspec(thread) or the
// pthreads API functions.

#if defined(R__HAS___THREAD)

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

