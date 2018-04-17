// $Id$
// Author: Sergey Linev   28/12/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFastCgi
#define ROOT_TFastCgi

#include "THttpEngine.h"

class TThread;

class TFastCgi : public THttpEngine {
protected:
   Int_t fSocket;       ///<! socket used by fastcgi
   Bool_t fDebugMode;   ///<! debug mode, may required for fastcgi debugging in other servers
   TString fTopName;    ///<! name of top item
   TThread *fThrd;      ///<! thread which takes requests, can be many later
   Bool_t fTerminating; ///<! set when http server wants to terminate all engines

   virtual void Terminate() { fTerminating = kTRUE; }

public:
   TFastCgi();
   virtual ~TFastCgi();

   Int_t GetSocket() const { return fSocket; }

   virtual Bool_t Create(const char *args);

   static void *run_func(void *);
};

#endif
