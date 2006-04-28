// @(#)root/cintex:$Name:  $:$Id: Cintex.h,v 1.3 2005/11/17 14:12:33 roiser Exp $
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
  namespace Cintex {
    
    struct Callback : public ROOT::Reflex::ICallback {
      virtual void operator () ( const ROOT::Reflex::Type& t );
      virtual void operator () ( const ROOT::Reflex::Member& m );
    };

    typedef TClass* (*ROOTCreator)( ROOT::Reflex::Type, ROOT::TGenericClassInfo* );
 
    class Cintex {
    public:
      Cintex();
      ~Cintex();
      static void Enable();
      static void SetROOTCreator(ROOTCreator);
      static ROOTCreator GetROOTCreator();
      static int  Debug();
      static void SetDebug(int);
      static bool PropagateClassTypedefs();
      static void SetPropagateClassTypedefs(bool);
    private:
      static Cintex& Instance();
      Callback*     fCallback;
      ROOTCreator   fRootcreator;
      int           fDbglevel;
      bool          fPropagateClassTypedefs;
      bool          fEnabled;
    };

  }
}
#endif // ROOT_Cintex_Cintex
