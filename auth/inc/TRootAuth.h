// @(#)root/auth:$Id$
// Author: Gerardo Ganis   08/07/05

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootAuth
#define ROOT_TRootAuth


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootAuth                                                            //
//                                                                      //
// TVirtualAuth implementation based on the old client authentication   //
// code.                                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualAuth
#include "TVirtualAuth.h"
#endif
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TSecContext;
class TSocket;

class TRootAuth : public TVirtualAuth {

public:
   TRootAuth() { }
   virtual ~TRootAuth() { }

   TSecContext *Authenticate(TSocket *, const char *host,
                             const char *user, Option_t *options = "");
   Int_t        ClientVersion();
   void         ErrorMsg(const char *where, Int_t ecode = -1);
   const char  *Name() { return "Root"; }

   ClassDef(TRootAuth,0)  // client auth interface
};

#endif
