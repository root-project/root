// @(#)root/gui:$Id$
// Author: Fons Rademakers   15/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TRootApplication
#define ROOT_TRootApplication


#include "TApplicationImp.h"


class TGClient;


class TRootApplication : public TApplicationImp {

private:
   TGClient    *fClient;        // pointer to the client environment
   char        *fDisplay;       // display server to connect to

   TRootApplication() { fClient = nullptr; fDisplay = nullptr; }
   void GetOptions(Int_t *argc, char **argv);

public:
   TRootApplication(const char *appClassName, Int_t *argc, char **argv);
   virtual ~TRootApplication();

   TGClient *Client() const { return fClient; }

   void    Show() override {}
   void    Hide() override {}
   void    Iconify() override {}
   Bool_t  IsCmdThread() override;
   void    Init() override {}
   void    Open() override {}
   void    Raise() override {}
   void    Lower() override {}

   ClassDefOverride(TRootApplication,0)  // ROOT native GUI application environment
};

#endif
