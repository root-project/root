// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Cintex_Cintex
#define ROOT_Cintex_Cintex

#include "Reflex/Callback.h"
#include "Reflex/Type.h"
#include "Reflex/Member.h"

namespace ROOT {
class TGenericClassInfo;
} // namespace ROOT
class TClass;

//class Cintex {
//public:
//static void Enable();
//};

namespace ROOT {
namespace Cintex {


//______________________________________________________________________________
class Callback : public ROOT::Reflex::ICallback {
public:
   virtual void operator()(const ROOT::Reflex::Type&);
   virtual void operator()(const ROOT::Reflex::Member&);
};

//______________________________________________________________________________
typedef TClass* (*ROOTCreator_t)(ROOT::Reflex::Type, ROOT::TGenericClassInfo*);

//______________________________________________________________________________
class Cintex {
   // Master Cintex controller, singleton.
private: // Static Data Members.
   static Cintex& Instance();
public: // Static Interface.
   static void Enable();
   static void SetROOTCreator(ROOTCreator_t);
   static ROOTCreator_t GetROOTCreator();
   static int Debug();
   static void SetDebug(int);
   static bool PropagateClassTypedefs();
   static void SetPropagateClassTypedefs(bool);
   static bool PropagateClassEnums();
   static void SetPropagateClassEnums(bool);
   static void Default_CreateClass(const char* name, TGenericClassInfo*);
private: // Data Members.
   Callback* fCallback;
   ROOTCreator_t fRootcreator;
   int fDbglevel;
   bool fPropagateClassTypedefs;
   bool fPropagateClassEnums;
   bool fEnabled;
public: // Public Interface.
   Cintex();
   ~Cintex();
};

} // namespace Cintex
} // namespace ROOT

using ROOT::Cintex::Cintex;

#endif // ROOT_Cintex_Cintex
