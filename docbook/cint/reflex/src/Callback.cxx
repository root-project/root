// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
# define REFLEX_BUILD
#endif

#include "Reflex/Kernel.h"
#include "Reflex/Callback.h"
#include <list>

// typedef std::list<Reflex::ICallback*> CbList;
// On RH7.3 the callback list is destructed before clients Get
// the chance to uninstall their callbacks. So, let's build some
// protection.
class CbList: public std::list<Reflex::ICallback*> {
public:
   CbList(): fAlive(true) {}

   ~CbList() { fAlive = false; }

   bool
   IsAlive() { return fAlive; }

private:
   typedef bool Bool_t;
   Bool_t fAlive;
};

//------------------------------------------------------------------------------
static CbList&
sClassCallbacks() {
//------------------------------------------------------------------------------
// Wraper for static callback list.
   static CbList* m = 0;

   if (!m) {
      m = new CbList;
   }
   return *m;
}


//-------------------------------------------------------------------------------
void
Reflex::InstallClassCallback(Reflex::ICallback* cb) {
//-------------------------------------------------------------------------------
// Install a class callback.
   sClassCallbacks().push_back(cb);
}


//-------------------------------------------------------------------------------
void
Reflex::UninstallClassCallback(Reflex::ICallback* cb) {
//-------------------------------------------------------------------------------
// Uninstall a class callback.
   if (sClassCallbacks().IsAlive()) {
      sClassCallbacks().remove(cb);
   }
}


//-------------------------------------------------------------------------------
void
Reflex::FireClassCallback(const Reflex::Type& ty) {
//-------------------------------------------------------------------------------
// Activate a class callback.
   for (CbList::const_iterator i = sClassCallbacks().begin();
        i != sClassCallbacks().end(); i++) {
      (**i)(ty);
   }
}


//-------------------------------------------------------------------------------
void
Reflex::FireFunctionCallback(const Reflex::Member& mem) {
//-------------------------------------------------------------------------------
// Activate a function callback.
   for (CbList::const_iterator i = sClassCallbacks().begin();
        i != sClassCallbacks().end(); i++) {
      (**i)(mem);
   }
}
