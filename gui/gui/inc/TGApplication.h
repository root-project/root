// @(#)root/gui:$Id$
// Author: Guy Barrand   30/05/2001

/*************************************************************************
 * Copyright (C) 2001, Guy Barrand.                                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGApplication
#define ROOT_TGApplication


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGApplication                                                        //
//                                                                      //
// This class initialize the ROOT GUI toolkit.                          //
// This class must be instantiated exactly once in any given            //
// application.                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TApplication.h"

class TGClient;


class TGApplication : public TApplication {

private:
   TString        fDisplay;           // display server to connect to
   TGClient      *fClient{nullptr};   // pointer to the client environment

protected:
   TGApplication() : TApplication() { }
   virtual void LoadGraphicsLibs();

public:
   TGApplication(const char *appClassName,
                 Int_t *argc, char **argv,
                 void *options = nullptr, Int_t numOptions = 0);
   virtual ~TGApplication();

   virtual void GetOptions(Int_t *argc, char **argv);

   ClassDef(TGApplication,0)  //GUI application singleton
};

#endif
