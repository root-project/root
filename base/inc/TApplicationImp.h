// @(#)root/base:$Name$:$Id$
// Author: Fons Rademakers   22/12/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TApplicationImp
#define ROOT_TApplicationImp

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TApplicationImp                                                      //
//                                                                      //
// ABC describing GUI independent application implementation protocol.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TGWin32Command;


class TApplicationImp {

public:
   TApplicationImp() { }
   TApplicationImp(const char *appClassName, int *argc, char **argv,
                   void *options, int numOptions);
   virtual ~TApplicationImp() { }

   virtual char   *ApplicationName() const { return 0; }
   virtual void    Show() { }
   virtual void    Hide() { }
   virtual void    Iconify() { }
   virtual Bool_t  IsCmdThread() { return kTRUE; } // by default (for UNIX) ROOT is a single thread application
   virtual void    Init() { }
   virtual void    Open() { }
   virtual void    Raise() { }
   virtual void    Lower() { }
   virtual Int_t   ExecCommand(TGWin32Command *code, Bool_t synch);

   ClassDef(TApplicationImp,0)  //ABC describing application implementation protocol
};

inline TApplicationImp::TApplicationImp(const char *, int *, char **,
                                        void *, int) { }
inline Int_t TApplicationImp::ExecCommand(TGWin32Command *, Bool_t) { return 0; }

#endif
