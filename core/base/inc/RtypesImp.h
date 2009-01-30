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

#include "TInterpreter.h"
#include "TClassEdit.h"

namespace ROOT {
   inline void GenericShowMembers(const char *topClassName,
                                  void *obj, TMemberInspector &R__insp,
                                  char *R__parent,
                                  bool transientMember)
   {
      // This could be faster if we implemented this either as a templated
      // function or by rootcint-generated code using the typeid (i.e. the
      // difference is a lookup in a TList instead of in a map).

      // To avoid a spurrious error message in case the data member is
      // transient and does not have a dictionary we check first.
      if (transientMember) {
         if (!TClassEdit::IsSTLCont(topClassName)) {
            ClassInfo_t *b = gInterpreter->ClassInfo_Factory(topClassName);
            Bool_t isloaded = gInterpreter->ClassInfo_IsLoaded(b);
            gInterpreter->ClassInfo_Delete(b);
            if (!isloaded) return;
         }
      }

      TClass *top = TClass::GetClass(topClassName);
      if (top) {
         top->CallShowMembers(obj, R__insp, R__parent);
      } else {
         // This might be worth an error message
      }
   }

  class TOperatorNewHelper { };
}

// This is to provide a placement operator new on all platforms
inline void *operator new(size_t /*size*/, ROOT::TOperatorNewHelper *p)
{
   return((void*)p);
}

#ifdef R__PLACEMENTDELETE
// this should never be used but help quiet down some compiler!
inline void operator delete(void*, ROOT::TOperatorNewHelper*) { }
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

#endif
