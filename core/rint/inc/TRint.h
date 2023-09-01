// @(#)root/rint:$Id$
// Author: Rene Brun   17/02/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TRint
#define ROOT_TRint

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Rint                                                                 //
//                                                                      //
// Rint is the ROOT Interactive Interface. It allows interactive access //
// to the ROOT system via a C++ interpreter.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TApplication.h"
#include "TString.h"

class TFileHandler;


class TRint : public TApplication {

private:
   Int_t         fNcmd;               // command history number
   TString       fDefaultPrompt;      // default prompt: "root [%d] "
   TString       fNonContinuePrompt;  // default prompt before continue prompt was set
   char          fPrompt[64];         // interpreter prompt
   Bool_t        fInterrupt;          // if true macro execution will be stopped
   Int_t         fCaughtSignal;       // TRint just caught a signal
   TFileHandler *fInputHandler;       // terminal input handler
   Bool_t        fBackslashContinue{};// whether the last line ended with '\'

   TRint(const TRint&) = delete;
   TRint& operator=(const TRint&) = delete;

   void    ExecLogon();
   Longptr_t ProcessRemote(const char *line, Int_t *error = nullptr) override;
   Longptr_t ProcessLineNr(const char* filestem, const char *line, Int_t *error = nullptr);

public:
   TRint(const char *appClassName, Int_t *argc, char **argv, void *options = nullptr, Int_t numOptions = 0,
         Bool_t noLogo = kFALSE, Bool_t exitOnUnknownArgs = kFALSE);
   virtual             ~TRint();
   virtual char       *GetPrompt();
   virtual const char *SetPrompt(const char *newPrompt);
           void        SetEchoMode(Bool_t mode) override;
           void        HandleException(Int_t sig) override;
           Bool_t      HandleTermInput() override;
   virtual void        PrintLogo(Bool_t lite = kFALSE);
           void        Run(Bool_t retrn = kFALSE) override;
   virtual void        Terminate(int status) override;
           void        Interrupt() { fInterrupt = kTRUE; }
           Int_t       TabCompletionHook(char *buf, int *pLoc, std::ostream& out) override;

   TFileHandler       *GetInputHandler() { return fInputHandler; }

   ClassDefOverride(TRint,0);  //ROOT Interactive Application Interface
};

#endif
