// @(#)root/auth:$Id$
// Author: G. Ganis, Nov 2006

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAFS
#define ROOT_TAFS


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAFS                                                                 //
//                                                                      //
// Utility class to acquire and handle an AFS tokens.                   //
// Interface to libTAFS.so.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TPluginHandler;

class TAFS : public TObject {
private:
   void                   *fToken;  // AFS token

   static Bool_t           fgUsePwdDialog;
   static TPluginHandler  *fgPasswdDialog;

public:
   TAFS(const char *fn = 0, const char *usr = 0, int life = -1);
   virtual ~TAFS();

   Int_t  Verify();

   static void  SetUsePwdDialog(Bool_t on = kTRUE);

   ClassDef(TAFS,0)  //AFS wrapper class
};

#endif

