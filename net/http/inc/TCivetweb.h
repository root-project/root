// $Id$
// Author: Sergey Linev   21/12/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCivetweb
#define ROOT_TCivetweb

#include "THttpEngine.h"
#include "TString.h"

class TCivetweb : public THttpEngine {
protected:
   void *fCtx;          ///<! civetweb context
   void *fCallbacks;    ///<! call-back table for civetweb webserver
   TString fTopName;    ///<! name of top item
   Bool_t fDebug;       ///<! debug mode
   Bool_t fTerminating; ///<! server doing shutdown and not react on requests

   virtual void Terminate() { fTerminating = kTRUE; }

public:
   TCivetweb();
   virtual ~TCivetweb();

   virtual Bool_t Create(const char *args);

   const char *GetTopName() const { return fTopName.Data(); }

   Bool_t IsDebugMode() const { return fDebug; }

   Bool_t IsTerminating() const { return fTerminating; }

   Int_t ProcessLog(const char *message);

   ClassDef(TCivetweb, 0) // http server implementation, based on civetweb embedded server
};

#endif
