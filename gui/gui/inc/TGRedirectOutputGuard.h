// @(#)root/gui:$Id$
// Author: G. Ganis   10/10/2005

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGRedirectOutputGuard
#define ROOT_TGRedirectOutputGuard

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGRedirectOutputGuard                                                //
//                                                                      //
// This class provides output redirection to a TGTextView in guaranteed //
// exception safe way. Use like this:                                   //
// {                                                                    //
//    TGRedirectOutputGuard guard(textview);                            //
//    ... // do something                                               //
//    guard.Update();                                                   //
//    ... // do something else                                          //
// }                                                                    //
// when guard goes out of scope, Update() is called to flush what left  //
// on the screed and the output is automatically redirected again to    //
// the standard units.                                                  //
// The exception mechanism takes care of calling the dtors              //
// of local objects so it is exception safe.                            //
// Optionally the output can also be saved into a file:                 //
// {                                                                    //
//    TGRedirectOutputGuard guard(textview, file, mode);                //
//    ... // do something                                               //
// }                                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TString.h"

class TGTextView;

class TGRedirectOutputGuard {

private:
   TString      fLogFile;
   Bool_t       fTmpFile;
   TGTextView  *fTextView;
   FILE        *fLogFileRead;

private:
   TGRedirectOutputGuard(const TGRedirectOutputGuard&) = delete;
   TGRedirectOutputGuard &operator=(const TGRedirectOutputGuard&) = delete;

public:
   TGRedirectOutputGuard(TGTextView *tv,
                         const char *flog = nullptr, const char *mode = "a");
   virtual ~TGRedirectOutputGuard();

   void Update(); // Update window with file content

   ClassDef(TGRedirectOutputGuard,0)  // Exception safe output redirection
};

#endif
