// Author: Sergey Linev   21/12/2013

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCivetweb
#define ROOT_TCivetweb

#include "THttpEngine.h"
#include "TString.h"

#include "../civetweb/civetweb.h"

class TCivetweb : public THttpEngine {
protected:
   struct mg_context *fCtx{nullptr}; ///<! civetweb context
   struct mg_callbacks fCallbacks;  ///<! call-back table for civetweb webserver
   TString fTopName;            ///<! name of top item
   Bool_t fDebug{kFALSE};       ///<! debug mode
   Bool_t fTerminating{kFALSE}; ///<! server doing shutdown and not react on requests
   Bool_t fOnlySecured{kFALSE}; ///<! if server should run only https protocol
   Int_t fMaxAge{3600};         ///<! max-age parameter
   Bool_t fWinSymLinks{kTRUE};  ///<! resolve symbolic links on Windows

   void Terminate() override { fTerminating = kTRUE; }

   Bool_t IsSecured() const { return fOnlySecured; }

public:
   TCivetweb(Bool_t only_secured = kFALSE);
   virtual ~TCivetweb();

   Bool_t Create(const char *args) override;

   const char *GetTopName() const { return fTopName.Data(); }

   Bool_t IsDebugMode() const { return fDebug; }

   Bool_t IsTerminating() const { return fTerminating; }

   Bool_t IsWinSymLinks() const { return fWinSymLinks; }

   Int_t ProcessLog(const char *message);

   Int_t GetMaxAge() const { return fMaxAge; }
};

#endif
