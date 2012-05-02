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

// ROOT forward declarations
namespace ROOT { class TGenericClassInfo; }
class TClass;


namespace ROOT {
  class TGenericClassInfo;
  namespace Cintex {
    
    class Callback : public ROOT::Reflex::ICallback {
     public:
      virtual void operator () ( const ROOT::Reflex::Type& t );
      virtual void operator () ( const ROOT::Reflex::Member& m );
    };

    typedef TClass* (*ROOTCreator_t)( ROOT::Reflex::Type, ROOT::TGenericClassInfo* );
 
    class Cintex {
    private:
       Cintex(const Cintex&); // not implemented
       Cintex& operator=(const Cintex&); // not implemented
    public:
      Cintex();
      ~Cintex();
      static void Enable();
      static void SetROOTCreator(ROOTCreator_t);
      static ROOTCreator_t GetROOTCreator();
      static int  Debug();
      static void SetDebug(int);
      static bool PropagateClassTypedefs();
      static void SetPropagateClassTypedefs(bool);
      static bool PropagateClassEnums();
      static void SetPropagateClassEnums(bool);
      static void Default_CreateClass(const char* name, TGenericClassInfo* gci);

    private:
      static Cintex& Instance();
      Callback*     fCallback;
      ROOTCreator_t fRootcreator;
      int           fDbglevel;
      bool          fPropagateClassTypedefs;
      bool          fPropagateClassEnums;
      bool          fEnabled;
    };

  }
}
#endif // ROOT_Cintex_Cintex
