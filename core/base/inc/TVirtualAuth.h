// Author: G. Ganis   08/07/05

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualAuth
#define ROOT_TVirtualAuth

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualAuth                                                         //
//                                                                      //
// Abstract interface for client authentication code.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"

class TSecContext;
class TSocket;

class TVirtualAuth {

public:
   TVirtualAuth() {}
   virtual ~TVirtualAuth() {}

   virtual TSecContext *Authenticate(TSocket *, const char *host,
                                     const char *user, Option_t *options) = 0;
   virtual Int_t        ClientVersion() = 0;
   virtual void         ErrorMsg(const char *where, Int_t ecode) = 0;
   virtual const char  *Name() = 0;

   ClassDef(TVirtualAuth,0)  // client auth interface
};

#endif
