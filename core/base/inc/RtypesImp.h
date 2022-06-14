// @(#)root/base:$Id$
// Author: Philippe Canal   23/2/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RtypesImp
#define ROOT_RtypesImp

#ifndef G__DICTIONARY
#error RtypesImp.h should only be included by ROOT dictionaries.
#endif

#include "TMemberInspector.h"
#include "TError.h"

namespace ROOT {
namespace Internal {
   inline void GenericShowMembers(const char *topClassName,
                                  void *obj, TMemberInspector &R__insp,
                                  bool transientMember)
   {
      Warning("ROOT::Internal::GenericShowMembers", "Please regenerate your dictionaries!");
      R__insp.GenericShowMembers(topClassName, obj, transientMember);
   }

  class TOperatorNewHelper { };
}
}

// This is to provide a placement operator new on all platforms
inline void *operator new(size_t /*size*/, ROOT::Internal::TOperatorNewHelper *p)
{
   return((void*)p);
}

#ifdef R__PLACEMENTDELETE
// this should never be used but help quiet down some compiler!
inline void operator delete(void*, ROOT::Internal::TOperatorNewHelper*) { }
#endif

// The STL GenerateInitInstance are not unique and hence are declared static
// (not accessible outside the dictionary and not linker error for duplicate)
#if defined(__CINT__)
#define RootStlStreamer(name,STREAMER)
#else
#define RootStlStreamer(name,STREAMER)                               \
namespace ROOT {                                                     \
   static TGenericClassInfo *GenerateInitInstance(const name*);      \
   static Short_t _R__UNIQUE_(R__dummyStreamer) =                    \
           GenerateInitInstance((name*)0x0)->SetStreamer(STREAMER);  \
   R__UseDummy(_R__UNIQUE_(R__dummyStreamer));                       \
}
#endif


#if defined(__cplusplus)
/* Helper class to avoid compiler warning about casting function pointer
** to void pointer.
*/
class TFunc2void {
   typedef void (*funcptr_t)();

   union funcptr_and_voidptr {

      funcptr_and_voidptr(void *val) : _read(val) {}

      void *_read;
      funcptr_t _write;
   };

   funcptr_and_voidptr _tmp;
public:
   template <typename T>
   TFunc2void( T vfp ) : _tmp(0) {
      _tmp._write = ( funcptr_t )vfp;
   }

   operator void* () const {
      return _tmp._read;
   }
};
#else
typedef union {
   void *_read;
   void (*_write)();
} funcptr_and_voidptr_t;
#endif

#endif
