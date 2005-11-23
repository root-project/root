// @(#)root/reflex:$Name:  $:$Id: Callback.h,v 1.2 2005/11/03 15:24:40 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Callback
#define ROOT_Reflex_Callback

// Include files

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
      class ICallback {
      
      public:

         /** destructor */
         virtual ~ICallback() {}

         /**
          * operator call (virtual)
          */
         virtual void operator () ( const Type & ) = 0;
         virtual void operator () ( const Member & ) = 0;

      }; // class ICallback
    
    
      void InstallClassCallback( ICallback * cb );
      void UninstallClassCallback( ICallback * cb );
      void FireClassCallback( const Type & );
      void FireFunctionCallback( const Member & );
    
   } // namespace Reflex
} // namespace ROOT



#endif // ROOT_Reflex_Callback
