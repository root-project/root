// @(#)root/base:$Name:  $:$Id: RtypesImp.h,v 1.7 2002/05/10 21:32:08 brun Exp $
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
   inline void GenericShowMembers(const char *topClassName,
                                  void *obj, TMemberInspector &R__insp,
                                  char *R__parent)
   {
      // This could be faster if we implemented this either as a templated
      // function or by rootcint-generated code using the typeid (i.e. the
      // difference is a lookup a in TList instead of in a map).
      TClass *top = gROOT->GetClass(topClassName);
      if (top) {
         ShowMembersFunc_t show = top->GetShowMembersWrapper();
         if (show) show(obj, R__insp, R__parent);
      }
   }
}

#endif
