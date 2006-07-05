// @(#)root/reflex:$Name: HEAD $:$Id: Callback.h,v 1.7 2006/03/13 21:34:36 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Callback
#define ROOT_Reflex_Callback

// Include files
#include "Reflex/Kernel.h"

namespace ROOT {
   namespace Reflex {

      // forward declarations
      class Type;
      class Member;

      /** 
       * @class Callback Callback.h Reflex/Callback.h
       * @author Pere Mato
       * @date 12/11/2004
       * @ingroup Ref
       */
      class RFLX_API ICallback {
      
      public:

         /** constructor */
         ICallback() {}

         /** destructor */
         virtual ~ICallback() {}

         /**
          * operator call (virtual)
          */
         virtual void operator () ( const Type & ) = 0;
         virtual void operator () ( const Member & ) = 0;

      }; // class ICallback
    
    
      RFLX_API void InstallClassCallback( ICallback * cb );
      RFLX_API void UninstallClassCallback( ICallback * cb );
      RFLX_API void FireClassCallback( const Type & );
      RFLX_API void FireFunctionCallback( const Member & );
    
   } // namespace Reflex
} // namespace ROOT



#endif // ROOT_Reflex_Callback
