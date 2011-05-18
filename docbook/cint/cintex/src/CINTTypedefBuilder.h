// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Cintex_CINTTypdefBuilder
#define ROOT_Cintex_CINTTypdefBuilder

#include "Reflex/Type.h"

/*
 *   Cintex namespace declaration
 */
namespace ROOT {  namespace Cintex {

      /*  @class CINTTypedefBuilder CINTTypedefBuilder.h
       *
       *    @author  M.Frank
       *    @version 1.0
       *    @date    10/04/2005
       */
      class CINTTypedefBuilder {
      public:
         // Declare typedef to CINT
         static int Setup(const ROOT::Reflex::Type& t);
         static void Set( const char* name, const char* value);
      };
   }}

#endif // ROOT_Cintex_CINTTypdefBuilder
