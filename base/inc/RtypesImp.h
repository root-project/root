// @(#)root/base:$Name:$:$Id:$
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

namespace ROOT {
   const InitBehavior *DefineBehavior(void * /*parent_type*/,
                                      void * /*actual_type*/);

#if 0
   template <class T> GenericClassInfo &GenerateInitInstance(const T*) {
      ::Warning("GenerateInitInstance", "No Dictionary Created for this class");
      static GenericClassInfo nullClassInfo("NoDict",0,"",0,typeid(T),0,0,0,0,0,0);
      return nullClassInfo;
   }
#endif

   inline void GenericShowMembers(const char *topClassName,
                                  void *obj, TMemberInspector &R__insp,
                                  char *R__parent) {
      // This could be faster if we implemented this either as a templated
      // function or by rootcint-generated code using the typeid (i.e. the
      // difference is a lookup a in TList instead of in a map.
      TClass *top = gROOT->GetClass(topClassName);
      if (top) {
         ShowMembersFunc_t show = top->GetShowMembersWrapper();
         if (show) show(obj, R__insp, R__parent);
      }
   }


#if 0
   template <class T> const InitBehavior *ClassInfo<T>::fgAction       = 0;
   template <class T> TClass             *ClassInfo<T>::fgClass        = 0;
   template <class T> const char         *ClassInfo<T>::fgClassName    = 0;
   template <class T> Int_t               ClassInfo<T>::fgVersion      = 0;
   template <class T> const char         *ClassInfo<T>::fgImplFileName = 0;
   template <class T> Int_t               ClassInfo<T>::fgImplFileLine = 0;
   template <class T> const char         *ClassInfo<T>::fgDeclFileName = 0;
   template <class T> Int_t               ClassInfo<T>::fgDeclFileLine = 0;
   template <class T> typename ClassInfo<T>::ShowMembersFunc_t  ClassInfo<T>::fgShowMembers = 0;
#endif

}

#endif
