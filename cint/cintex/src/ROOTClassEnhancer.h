// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Cintex_ROOTClassEnhancer
#define ROOT_Cintex_ROOTClassEnhancer

#include "Reflex/Type.h"

// ROOT classes forward declarations
class TClass;
class TMemberInspector;

namespace ROOT {
   class TGenericClassInfo;

   namespace Cintex {

      class ROOTClassEnhancer {
      public:
         ROOTClassEnhancer(const Reflex::Type& TypeNth);
         ~ROOTClassEnhancer();
         void Setup(void);
         void CreateInfo(void);
         static void CreateClassForNamespace(const std::string&);
         static TClass* Default_CreateClass(Reflex::Type typ, ROOT::TGenericClassInfo* info);

      private:
         ROOT::Reflex::Type  fClass;
         std::string         fName;
         void*               fEnhancerinfo;
      };
    
   }
}

#endif // ROOT_Cintex_ROOTClassEnhancer
