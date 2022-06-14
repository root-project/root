// @(#)root/auth:$Id$
// Author: G. Ganis   08/07/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootSecContext
#define ROOT_TRootSecContext


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootSecContext                                                      //
//                                                                      //
// Special implementation of TSecContext                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAuthenticate.h"
#include "TSecContext.h"

class TRootSecContext : public TSecContext {

private:
   Int_t        fRSAKey;              // Type of RSA key used

   Bool_t       CleanupSecContext(Bool_t all) override;

public:

   TRootSecContext(const char *url, Int_t meth, Int_t offset,
               const char *id, const char *token,
               TDatime expdate = kROOTTZERO, void *ctx = 0, Int_t key = 1);
   TRootSecContext(const char *user, const char *host, Int_t meth, Int_t offset,
               const char *id, const char *token,
               TDatime expdate = kROOTTZERO, void *ctx = 0, Int_t key = 1);
   virtual    ~TRootSecContext();

   const char *AsString(TString &out) override;

   void        DeActivate(Option_t *opt = "CR") override;
   Int_t       GetRSAKey()  const { return fRSAKey; }

   void        Print(Option_t *option = "F") const override;

   ClassDefOverride(TRootSecContext,0)  // Class providing host specific authentication information
};

#endif
