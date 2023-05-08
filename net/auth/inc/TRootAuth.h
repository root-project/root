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

#include "TVirtualAuth.h"
#include "Rtypes.h"

class TSecContext;
class TSocket;

class TRootAuth : public TVirtualAuth {

public:
   TRootAuth() {}
   virtual ~TRootAuth() {}

   TSecContext *Authenticate(TSocket *, const char *host,
                             const char *user, Option_t *options = "") override;
   Int_t        ClientVersion() override;
   void         ErrorMsg(const char *where, Int_t ecode = -1) override;
   const char  *Name() override { return "Root"; }

   ClassDefOverride(TRootAuth,0)  // client auth interface
};

#endif
