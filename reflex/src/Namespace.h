// @(#)root/reflex:$Name: HEAD $:$Id: Namespace.h,v 1.4 2006/03/06 12:51:46 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Namespace
#define ROOT_Reflex_Namespace

// Include files
#include "Reflex/Scope.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations
      class Member;
      class Scope;

      /**
       * @class Namespace Namespace.h Reflex/Namespace.h
       * @author Stefan Roiser
       * @date 24/11/2003
       * @ingroup Ref
       */
      class Namespace : public ScopeBase {

      public:

         /** default constructor */
         Namespace( const char * scop );


         /** destructor */
         virtual ~Namespace() {}


         /**
          * function for initialisation of the global namespace
          */
         static void InitGlobalNamespace();

      private:

         /** constructor for initialisation of the global namespace */
         Namespace();

      }; // class Namespace
   } //namespace Reflex
} //namespace ROOT

#endif // ROOT_Reflex_Namespace
