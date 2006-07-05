// @(#)root/reflex:$Name: HEAD $:$Id: Callback.cxx,v 1.7 2006/07/04 15:02:55 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
#define REFLEX_BUILD
#endif

#include "Reflex/Kernel.h"
#include "Reflex/Callback.h"
#include <list>

// typedef std::list<ROOT::Reflex::ICallback*> CbList;
// On RH7.3 the callback list is destructed before clients Get
// the chance to uninstall their callbacks. So, let's build some
// protection. 
class  CbList : public std::list<ROOT::Reflex::ICallback*> 
{
public:
   CbList() : fAlive(true) {}
   ~CbList() { fAlive = false; }
   bool IsAlive() { return fAlive; }
private:
   typedef bool Bool_t;
   Bool_t fAlive;
};

//------------------------------------------------------------------------------
static CbList & sClassCallbacks() {
//------------------------------------------------------------------------------
// Wraper for static callback list.
   static CbList m;
   return m;
}

//-------------------------------------------------------------------------------
void ROOT::Reflex::InstallClassCallback( ROOT::Reflex::ICallback * cb ) {
//-------------------------------------------------------------------------------
// Install a class callback.
   sClassCallbacks().push_back( cb );
}

//-------------------------------------------------------------------------------
void ROOT::Reflex::UninstallClassCallback( ROOT::Reflex::ICallback * cb ) {
//-------------------------------------------------------------------------------
// Uninstall a class callback.
   if( sClassCallbacks().IsAlive() ) {
      sClassCallbacks().remove( cb );
   }
}

//-------------------------------------------------------------------------------
void ROOT::Reflex::FireClassCallback( const ROOT::Reflex::Type & ty ) {
//-------------------------------------------------------------------------------
// Activate a class callback.
   for ( CbList::const_iterator i = sClassCallbacks().begin(); 
         i != sClassCallbacks().end(); i++ ) {
      (**i)(ty);
   }
}

//-------------------------------------------------------------------------------
void ROOT::Reflex::FireFunctionCallback( const ROOT::Reflex::Member & mem ) {
//-------------------------------------------------------------------------------
// Activate a function callback.
   for ( CbList::const_iterator i = sClassCallbacks().begin(); 
         i != sClassCallbacks().end(); i++ ) {
      (**i)(mem);
   }
}

