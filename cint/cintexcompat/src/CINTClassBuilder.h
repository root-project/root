// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Cintex_CINTClassBuilder
#define ROOT_Cintex_CINTClassBuilder

#include "Reflex/Type.h"
#include "CINTdefs.h"
#include "CINTFunctional.h"
#include <vector>
#include <map>

namespace ROOT {
namespace Cintex {

class CINTClassBuilder {
private: // Private Types
   typedef std::vector<std::pair<ROOT::Reflex::Base, int> > Bases;
   class CINTClassBuilders : public std::map<ROOT::Reflex::Type, CINTClassBuilder*>  {
   public:
      static CINTClassBuilders& Instance();
   private:
      CINTClassBuilders();
      ~CINTClassBuilders();
   };
private: // Private Static Members
   static void* fgFakeObject;
   static void* fgFakeAddress;
public: // Public Static Interface
   static CINTClassBuilder& Get(const ROOT::Reflex::Type&);
   static void Setup_memfunc_with_context(void*);
   static void Setup_memvar_with_context(void*);
private: // Private Data Members
   ROOT::Reflex::Type fClass;
   G__linked_taginfo* fTaginfo;
   std::string fName;
   bool fPending;
   FuncVoidPtr_t fSetup_memvar;
   FuncVoidPtr_t fSetup_memfunc;
   Bases* fBases;
private: // Private Interface
   CINTClassBuilder(const ROOT::Reflex::Type&);
public: // Public Interface
   ~CINTClassBuilder();
   void Setup(void);
   void Setup_environment(void); // NOT IMPLEMENTED
   void Setup_tagtable(void);
   void Setup_memfunc(void);
   void Setup_memvar(void);
   void Setup_inheritance();
   void Setup_inheritance(ROOT::Reflex::Object&);
   void Setup_typetable(void);
   const std::string& Name() const {
      return fName;
   }
   ROOT::Reflex::Type& TypeGet() {
      return fClass;
   }
   Bases* GetBases();
};

} // namespace Cintex
} // namespace ROOT

#endif // ROOT_Cintex_CINTClassBuilder
