// @(#)root/gui:$Name:  $:$Id: TRootApplication.h,v 1.1.1.1 2000/05/16 17:00:42 rdm Exp $
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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootApplication                                                     //
//                                                                      //
// This class create the ROOT native GUI version of the ROOT            //
// application environment. This in contrast the Win32 version.         //
// Once the native widgets work on Win32 this class can be folded into  //
// the TApplication class (since all graphic will go via TVirtualX).         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TApplicationImp
#include "TApplicationImp.h"
#endif


class TGClient;


class TRootApplication : public TApplicationImp {

private:
   TGClient    *fClient;        // pointer to the client environment
   char        *fDisplay;       // display server to conntect to

   TRootApplication() { fClient = 0; fDisplay = 0; }
   void GetOptions(Int_t *argc, char **argv);

public:
   TRootApplication(const char *appClassName, Int_t *argc, char **argv,
                    void *options = 0, Int_t numOptions = 0);
   virtual ~TRootApplication();

   TGClient     *Client() const { return fClient; }

   void    Show() { }
   void    Hide() { }
   void    Iconify() { }
   Bool_t  IsCmdThread() { return kTRUE; } // by default (for UNIX) ROOT is a single thread application
   void    Init() { }
   void    Open() { }
   void    Raise() { }
   void    Lower() { }

   ClassDef(TRootApplication,0)  // ROOT native GUI application environment
};

#endif
