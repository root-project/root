// @(#)root/base:$Name$:$Id$
// Author: Fons Rademakers   15/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TBrowserImp
#define ROOT_TBrowserImp

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBrowserImp                                                          //
//                                                                      //
// ABC describing GUI independent browser implementation protocol.      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TBrowser;


class TBrowserImp {

protected:
   TBrowser  *fBrowser;     //TBrowser associated with this implementation
   Bool_t     fShowCycles;  //Show object cycle numbers in browser

public:
   TBrowserImp(TBrowser *b=0) : fBrowser(b), fShowCycles(kFALSE) { }
   TBrowserImp(TBrowser *b, const char *title, UInt_t width, UInt_t height);
   TBrowserImp(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TBrowserImp() { }

   virtual void  Add(TObject *, const char *) { }
   virtual void  BrowseObj(TObject *) { }
   TBrowser     *Browser() const { return fBrowser; }
   virtual void  ExecuteDefaultAction(TObject *) { }
   virtual void  Iconify() { }
   virtual void  RecursiveRemove(TObject *) { }
   virtual void  Refresh(Bool_t = kFALSE) { }
   virtual void  Show() { }

   ClassDef(TBrowserImp,0)  //ABC describing browser implementation protocol
};

inline TBrowserImp::TBrowserImp(TBrowser *, const char *, UInt_t, UInt_t) { }
inline TBrowserImp::TBrowserImp(TBrowser *, const char *, Int_t, Int_t, UInt_t, UInt_t) { }

#endif
